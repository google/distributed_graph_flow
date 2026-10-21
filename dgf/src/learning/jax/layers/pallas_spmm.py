# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pallas TPU kernel for O(N)-memory Sparse-Dense Matrix Multiplication (SpMM).

Implements Y = A @ X where A is a sparse matrix in COO format (row_ids, col_ids)
of shape [num_rows, num_cols] and X is a dense node feature tensor of shape
[num_cols, ...] without materializing O(E * D) edge feature tensors in HBM.

Architecture:
1. **Hardware-Aware TPU Device Profiles (`TPUDeviceProfile`)**:
   - Auto-detects the active TPU architecture (`GLP`, `VLP`, `GFL`, `GF`, `VF`,
     `PF`) via `pltpu.get_tpu_info()` and scales VMEM
     working-set budgets (`128 MiB` on `GLP`/`VLP`, `64 MiB` on `GFL`/`GF`/`VF`,
     `16 MiB` on `PF`).
2. **Single-Sort On-TPU Degree-Bucketed Schedule (`FastCoSort`)**:
   - Sorts edges by destination node (`uint16` key when `num_rows < 65535`) and
     computes exact capped degrees (`1..9`) in a single fused padded slice
     window without `jnp.bincount` or `jnp.roll` kernel launches.
   - Co-sorts `((edge_cat << 20) | r_s, c_s)` and finds category offsets via
     `jnp.searchsorted`.
   - Inside `_make_cosort_spmm_kernel`, direct-write categories `1..8`
     accumulate all `K` source vectors in `float32` VREGs and write once to
     `out_ref[dst]` with zero destination VMEM reads. Boundary tuples and
     Category `9` (`deg >= 9`) carry running `float32` sums in VREGs and a
     1-node `carry_vmem` buffer, enabling single-pass (`g_col = 1`) `bfloat16`
     `D = 256` execution even on `64 MiB` VMEM TPUs (`GFL`/`GF`/`VF`) with
     exact `0.00e+00` `fp32`-accumulated parity.
3. **Large-N 2D Edge-Bucketed Manual-DMA Kernel (`spmm_large`)**:
   - Partitions source and destination nodes into VMEM-resident blocks (`b_src`,
     `b_dst <= 32768`), buckets edges by 2D `(dst_blk, src_blk)` tile ID using
     `uint8`/`uint16` radix keys, and executes a single Pallas kernel with
     burst DMAs and full `(8, 128)` VREG tiles.
4. **Direct-Edge VMEM Streaming (`_run_pallas_spmm_2d`)**:
   - Used for weighted SpMM (`values is not None`) and ultra-dense unscheduled
     edge scaling (`E > 10 * N`, up to `20.97M` edges) with strictly constant
     `O(N)` HBM memory and SALU-guarded collision-free unrolled fast paths.
"""

import collections
import dataclasses
import functools
import math
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

# Default scoped VMEM limit on 128 MiB VMEM TPUs (TPU v5e / VLP, TPU v6e / GLP).
# On 64 MiB VMEM TPUs (TPU7 / GFL, TPU7x / GF, TPU v5p / VF) and 16 MiB VMEM
# TPUs (TPU v4 / PF), `get_tpu_device_profile()` automatically scales these
# budgets to fit physical VMEM capacity.
_DEFAULT_VMEM_LIMIT_BYTES = 122 * 1024 * 1024
_DEFAULT_MAX_WORKING_SET_BYTES = 115 * 1024 * 1024
_MAX_WORKING_SET_BYTES = _DEFAULT_MAX_WORKING_SET_BYTES
_DEFAULT_EDGE_BLOCK_SIZE = 4096
_DEFAULT_LARGE_EDGE_BLOCK_SIZE = 8192
_DEFAULT_UNROLL = 4
_ZERO_CHUNK_NODES = 1024
_LARGE_ZERO_CHUNK_NODES = 256
_LARGE_STAGE_NODES = 256

_DEFAULT_COSORT_BLOCK_SIZE = 32768
_DEFAULT_COSORT_UNROLL = 8
_INTERPRET_COSORT_BLOCK_SIZE = 64
_INTERPRET_COSORT_UNROLL = 1

_DEFAULT_BUCKET_BLOCK_SIZE = 1024
_MAX_SCHEDULE_CACHE_SIZE = 64

_SINGLE_SCHED_CACHE: collections.OrderedDict[
    tuple[Any, ...],
    tuple[Any, Any, tuple[jax.Array, jax.Array, jax.Array]],
] = collections.OrderedDict()

_PAIR_SCHED_CACHE: collections.OrderedDict[
    tuple[Any, ...],
    tuple[
        Any,
        Any,
        tuple[
            tuple[jax.Array, jax.Array, jax.Array],
            tuple[jax.Array, jax.Array, jax.Array],
        ],
    ],
] = collections.OrderedDict()


@dataclasses.dataclass(frozen=True, kw_only=True)
class TPUDeviceProfile:
  """Hardware and kernel tuning profile for a specific TPU architecture.

  Attributes:
    name: Canonical short profile identifier (e.g. 'glp', 'gfl', 'gf', 'vlp',
      'vf', 'pf', 'cpu_interpret').
    device_kind: JAX device_kind string (e.g. 'TPU v6 lite', 'TPU7', 'TPU7x',
      'TPU v5 lite', 'TPU v5', 'TPU v4').
    generation: TPU hardware generation (4, 5, 6, 7, 8).
    tensor_cores_per_chip: Physical TensorCores per chip (1 for lite chips, 2
      for dual-core fish chips).
    num_lanes: Vector register lane count (128).
    num_sublanes: Vector register sublane count (8).
    mxu_column_size: MXU systolic array width (128 on Gen <=5, 256 on Gen >=6).
    vmem_capacity_bytes: Physical VMEM capacity per TensorCore in bytes.
    smem_capacity_bytes: Physical SMEM capacity per TensorCore in bytes.
    hbm_capacity_bytes: Physical HBM capacity per TensorCore in bytes.
    mem_bw_bytes_per_second: Peak HBM bandwidth per TensorCore in bytes/s.
    vmem_limit_bytes: Scoped VMEM limit passed to pltpu.CompilerParams.
    max_working_set_bytes: Maximum VMEM working set (x_ref + y_ref + acc_ref)
      for 1D-grid pallas_spmm_kernel before column-chunking or spmm_large.
    large_avail_vmem_bytes: Available VMEM budget for 2D node-tiled spmm_large
      (x_vmem + acc_vmem + y_stage_vmem).
    max_large_chunk_dim: Maximum feature tile width per spmm_large call.
    default_edge_block_size: Default SMEM edge block size (multiple of 128).
    default_unroll: Default inner-loop unroll factor with store-to-load
      forwarding.
    bf16_segment_sum_uses_fp32: Whether XLA's native unsorted_segment_sum on
      bfloat16 promotes to fp32 (True on Gen < 7, False on Gen >= 7).
  """

  name: str
  device_kind: str
  generation: int
  tensor_cores_per_chip: int
  num_lanes: int = 128
  num_sublanes: int = 8
  mxu_column_size: int = 128
  vmem_capacity_bytes: int
  smem_capacity_bytes: int
  hbm_capacity_bytes: int
  mem_bw_bytes_per_second: int
  vmem_limit_bytes: int
  max_working_set_bytes: int
  large_avail_vmem_bytes: int
  max_large_chunk_dim: int = 1024
  default_edge_block_size: int = _DEFAULT_EDGE_BLOCK_SIZE
  default_unroll: int = _DEFAULT_UNROLL
  bf16_segment_sum_uses_fp32: bool = True


DEVICE_PROFILES: dict[str, TPUDeviceProfile] = {
    # GLP: GhostLite Pod / TPU v6e (Trillium) - 1 TensorCore/chip, 128 MiB VMEM
    'glp': TPUDeviceProfile(
        name='glp',
        device_kind='TPU v6 lite',
        generation=6,
        tensor_cores_per_chip=1,
        mxu_column_size=256,
        vmem_capacity_bytes=128 * 1024 * 1024,
        smem_capacity_bytes=1024 * 1024,
        hbm_capacity_bytes=34_400_000_000,
        mem_bw_bytes_per_second=1_640_000_000_000,
        vmem_limit_bytes=122 * 1024 * 1024,
        max_working_set_bytes=115 * 1024 * 1024,
        large_avail_vmem_bytes=116 * 1024 * 1024,
        max_large_chunk_dim=1024,
        default_edge_block_size=4096,
        default_unroll=4,
        bf16_segment_sum_uses_fp32=True,
    ),
    # VLP: ViperLite Pod / TPU v5e - 1 TensorCore/chip, 128 MiB VMEM
    'vlp': TPUDeviceProfile(
        name='vlp',
        device_kind='TPU v5 lite',
        generation=5,
        tensor_cores_per_chip=1,
        mxu_column_size=128,
        vmem_capacity_bytes=128 * 1024 * 1024,
        smem_capacity_bytes=1024 * 1024,
        hbm_capacity_bytes=17_200_000_000,
        mem_bw_bytes_per_second=820_000_000_000,
        vmem_limit_bytes=122 * 1024 * 1024,
        max_working_set_bytes=115 * 1024 * 1024,
        large_avail_vmem_bytes=116 * 1024 * 1024,
        max_large_chunk_dim=1024,
        default_edge_block_size=4096,
        default_unroll=4,
        bf16_segment_sum_uses_fp32=True,
    ),
    # GFL: GhostfishLite / TPU7 - 2 TensorCores/chip (split), 64 MiB VMEM/core
    'gfl': TPUDeviceProfile(
        name='gfl',
        device_kind='TPU7',
        generation=7,
        tensor_cores_per_chip=2,
        mxu_column_size=256,
        vmem_capacity_bytes=64 * 1024 * 1024,
        smem_capacity_bytes=1024 * 1024,
        hbm_capacity_bytes=103_000_000_000,
        mem_bw_bytes_per_second=3_700_000_000_000,
        vmem_limit_bytes=62 * 1024 * 1024,
        max_working_set_bytes=57 * 1024 * 1024,
        large_avail_vmem_bytes=54 * 1024 * 1024,
        max_large_chunk_dim=1024,
        default_edge_block_size=4096,
        default_unroll=4,
        bf16_segment_sum_uses_fp32=False,
    ),
    # GF: Ghostfish / TPU7x (Ironwood) - 2 TensorCores/chip, 64 MiB VMEM/core
    'gf': TPUDeviceProfile(
        name='gf',
        device_kind='TPU7x',
        generation=7,
        tensor_cores_per_chip=2,
        mxu_column_size=256,
        vmem_capacity_bytes=64 * 1024 * 1024,
        smem_capacity_bytes=1024 * 1024,
        hbm_capacity_bytes=103_000_000_000,
        mem_bw_bytes_per_second=3_700_000_000_000,
        vmem_limit_bytes=62 * 1024 * 1024,
        max_working_set_bytes=57 * 1024 * 1024,
        large_avail_vmem_bytes=54 * 1024 * 1024,
        max_large_chunk_dim=1024,
        default_edge_block_size=4096,
        default_unroll=4,
        bf16_segment_sum_uses_fp32=False,
    ),
    # VF: Viperfish / TPU v5p - 2 TensorCores/chip, 64 MiB VMEM/core
    'vf': TPUDeviceProfile(
        name='vf',
        device_kind='TPU v5',
        generation=5,
        tensor_cores_per_chip=2,
        mxu_column_size=128,
        vmem_capacity_bytes=64 * 1024 * 1024,
        smem_capacity_bytes=1024 * 1024,
        hbm_capacity_bytes=51_500_000_000,
        mem_bw_bytes_per_second=1_230_000_000_000,
        vmem_limit_bytes=62 * 1024 * 1024,
        max_working_set_bytes=57 * 1024 * 1024,
        large_avail_vmem_bytes=54 * 1024 * 1024,
        max_large_chunk_dim=1024,
        default_edge_block_size=4096,
        default_unroll=4,
        bf16_segment_sum_uses_fp32=True,
    ),
    # PF: Pufferfish / TPU v4 - 2 TensorCores/chip, 16 MiB VMEM/core
    'pf': TPUDeviceProfile(
        name='pf',
        device_kind='TPU v4',
        generation=4,
        tensor_cores_per_chip=2,
        mxu_column_size=128,
        vmem_capacity_bytes=16 * 1024 * 1024,
        smem_capacity_bytes=1024 * 1024,
        hbm_capacity_bytes=17_200_000_000,
        mem_bw_bytes_per_second=615_000_000_000,
        vmem_limit_bytes=15 * 1024 * 1024,
        max_working_set_bytes=13 * 1024 * 1024 + 512 * 1024,
        large_avail_vmem_bytes=11 * 1024 * 1024 + 512 * 1024,
        max_large_chunk_dim=256,
        default_edge_block_size=2048,
        default_unroll=4,
        bf16_segment_sum_uses_fp32=True,
    ),
}

_PROFILE_ALIASES: dict[str, str] = {
    'glp': 'glp',
    'ghostlite': 'glp',
    'ghostlite_pod': 'glp',
    'tpu v6 lite': 'glp',
    'tpu_v6e': 'glp',
    'v6e': 'glp',
    'vlp': 'vlp',
    'viperlite': 'vlp',
    'viperlite_pod': 'vlp',
    'viperfish_lite': 'vlp',
    'tpu v5 lite': 'vlp',
    'tpu_v5e': 'vlp',
    'v5e': 'vlp',
    'gfl': 'gfl',
    'ghostfishlite': 'gfl',
    'tpu7': 'gfl',
    'tpu_7': 'gfl',
    'gf': 'gf',
    'ghostfish': 'gf',
    'tpu7x': 'gf',
    'tpu_7x': 'gf',
    'vf': 'vf',
    'viperfish': 'vf',
    'tpu v5': 'vf',
    'tpu v5p': 'vf',
    'tpu_v5p': 'vf',
    'v5p': 'vf',
    'pf': 'pf',
    'pufferfish': 'pf',
    'tpu v4': 'pf',
    'tpu v4i': 'pf',
    'tpu_v4': 'pf',
    'v4': 'pf',
}


def _profile_from_vmem_and_specs(
    name: str,
    device_kind: str,
    generation: int,
    tensor_cores_per_chip: int,
    vmem_bytes: int,
    smem_bytes: int,
    hbm_bytes: int,
    mem_bw_bytes_per_second: int,
    num_lanes: int = 128,
    num_sublanes: int = 8,
    mxu_column_size: int = 128,
) -> TPUDeviceProfile:
  """Derives a safe TPUDeviceProfile from raw hardware capacities."""
  return TPUDeviceProfile(
      name=name,
      device_kind=device_kind,
      generation=generation,
      tensor_cores_per_chip=tensor_cores_per_chip,
      num_lanes=num_lanes,
      num_sublanes=num_sublanes,
      mxu_column_size=mxu_column_size,
      vmem_capacity_bytes=vmem_bytes,
      smem_capacity_bytes=smem_bytes,
      hbm_capacity_bytes=hbm_bytes,
      mem_bw_bytes_per_second=mem_bw_bytes_per_second,
      vmem_limit_bytes=(vmem_bytes * 61) // 64,
      max_working_set_bytes=(vmem_bytes * 57) // 64,
      large_avail_vmem_bytes=(vmem_bytes * 54) // 64,
      max_large_chunk_dim=1024 if vmem_bytes >= 64 * 1024 * 1024 else 256,
      default_edge_block_size=4096 if smem_bytes >= 256 * 1024 else 512,
      default_unroll=4,
      bf16_segment_sum_uses_fp32=generation < 7,
  )


def get_tpu_device_profile(
    device_profile: TPUDeviceProfile | str | None = None,
) -> TPUDeviceProfile:
  """Resolves or auto-detects the TPUDeviceProfile for the active device.

  Args:
    device_profile: Optional explicit `TPUDeviceProfile` instance or profile
      name/alias ('glp', 'gfl', 'gf', 'vlp', 'vf', 'pf').
      When None, queries `pltpu.get_tpu_info()` and `jax.devices()[0]` to
      select the hardware-matched profile automatically.

  Returns:
    The resolved `TPUDeviceProfile`.
  """
  if isinstance(device_profile, TPUDeviceProfile):
    return device_profile
  if isinstance(device_profile, str):
    key = device_profile.strip().lower()
    canonical = _PROFILE_ALIASES.get(key, key)
    if canonical not in DEVICE_PROFILES:
      raise ValueError(
          f'Unknown TPU device profile {device_profile!r}. Supported profiles: '
          f'{sorted(DEVICE_PROFILES.keys())} (aliases: '
          f'{sorted(_PROFILE_ALIASES.keys())}).'
      )
    return DEVICE_PROFILES[canonical]

  try:
    info = pltpu.get_tpu_info()
    dev_kind = jax.devices()[0].device_kind if jax.devices() else ''
    canonical = _PROFILE_ALIASES.get(dev_kind.strip().lower())
    if (
        canonical in DEVICE_PROFILES
        and DEVICE_PROFILES[canonical].vmem_capacity_bytes
        == int(info.vmem_capacity_bytes)
    ):
      return DEVICE_PROFILES[canonical]
    chip_ver = getattr(info, 'chip_version', None)
    cores_per_chip = int(
        getattr(chip_ver, 'num_physical_tensor_cores_per_chip', info.num_cores)
    )
    return _profile_from_vmem_and_specs(
        name=canonical or str(chip_ver or dev_kind or 'tpu_custom').lower(),
        device_kind=dev_kind or str(chip_ver or 'TPU'),
        generation=int(info.generation),
        tensor_cores_per_chip=cores_per_chip,
        vmem_bytes=int(info.vmem_capacity_bytes),
        smem_bytes=int(info.smem_capacity_bytes),
        hbm_bytes=int(info.hbm_capacity_bytes),
        mem_bw_bytes_per_second=int(info.mem_bw_bytes_per_second),
        num_lanes=int(info.num_lanes),
        num_sublanes=int(info.num_sublanes),
        mxu_column_size=int(info.mxu_column_size),
    )
  except Exception:  # pylint: disable=broad-except
    if jax.devices():
      dev_kind = getattr(jax.devices()[0], 'device_kind', '')
      canonical = _PROFILE_ALIASES.get(dev_kind.strip().lower())
      if canonical in DEVICE_PROFILES:
        return DEVICE_PROFILES[canonical]
    return DEVICE_PROFILES['glp']


def _effective_max_working_set_bytes(profile: TPUDeviceProfile) -> int:
  """Returns the active VMEM working-set limit, respecting test overrides."""
  if _MAX_WORKING_SET_BYTES != _DEFAULT_MAX_WORKING_SET_BYTES:
    return min(profile.max_working_set_bytes, _MAX_WORKING_SET_BYTES)
  return profile.max_working_set_bytes


def pick_3d_tile_shape(
    feat_dim: int, dtype: Any
) -> tuple[int, int, int]:
  """Selects hardware-aligned (m_tile, l_tile, d_pad) for a feature dimension.

  On TPU, native VREG tiles are (8, 128). For a 3D tensor of shape
  (N, m_tile, l_tile), Pallas/Mosaic infers tiling (1, m_tile, 128) whenever
  m_tile is in {1, 2, 4, 8} with m_tile >= packing (1 for float32, 2 for
  bfloat16) and l_tile is a multiple of 128. Because the leading node dimension
  has tiling 1, any single-node slice x_ref[pl.ds(u, 1)] of shape
  (1, m_tile, l_tile) is 100% hardware-aligned in VMEM and HBM.

  Args:
    feat_dim: Flattened feature dimension D.
    dtype: Element dtype (float32 or bfloat16).

  Returns:
    Tuple (m_tile, l_tile, d_pad) where d_pad = m_tile * l_tile >= feat_dim.
  """
  itemsize = jnp.dtype(dtype).itemsize
  packing = max(1, 4 // itemsize)
  if feat_dim == 256 and packing == 1:
    return 1, 256, 256
  valid_m_tiles = [m for m in (8, 4, 2, 1) if m >= packing]

  # First check if feat_dim is already an exact multiple of m_tile * 128.
  for m_tile in valid_m_tiles:
    if feat_dim % (m_tile * 128) == 0:
      l_tile = feat_dim // m_tile
      if l_tile <= 512:
        return m_tile, l_tile, feat_dim

  # Otherwise pick the smallest padded tile >= feat_dim.
  for m_tile in reversed(valid_m_tiles):
    if feat_dim <= m_tile * 128:
      return m_tile, 128, m_tile * 128

  m_tile = 8
  l_tile = ((feat_dim + (m_tile * 128) - 1) // (m_tile * 128)) * 128
  return m_tile, l_tile, m_tile * l_tile


@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def _build_tpu_spmm_schedule_jit(
    row_ids: jax.Array,
    col_ids: jax.Array,
    num_rows: int,
    num_cols: int,
    block_size: int = _DEFAULT_COSORT_BLOCK_SIZE,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """JIT-compiled degree-bucketed SpMM schedule builder with fused window."""
  e_len = row_ids.shape[0]
  num_blocks = (e_len + block_size - 1) // block_size
  e_aligned = num_blocks * block_size
  e_pad = e_aligned - e_len

  # Collapse self-loops `(num_rows - 1, num_cols - 1)` on the last node into a
  # scalar count `kept_pad_count` when `num_rows == num_cols` and the last node
  # only has self-loops (as in `FixedSizePadder`), skipping all 300K+ dummy
  # edges in the main blocks and accumulating them on the final block.
  if e_len >= 16384 and num_rows == num_cols and num_rows >= 64:
    is_last_row = row_ids == (num_rows - 1)
    is_last_self = is_last_row & (col_ids == (num_cols - 1))
    n_last_row = jnp.sum(is_last_row, dtype=jnp.int32)
    n_last_self = jnp.sum(is_last_self, dtype=jnp.int32)
    only_self = (n_last_row == n_last_self) & (n_last_self > 0)
    is_pad_skip = jnp.where(only_self, is_last_self, False)
    kept_pad_count = jnp.where(only_self, n_last_self, jnp.int32(0))
  else:
    is_pad_skip = jnp.bool_(False)
    kept_pad_count = jnp.int32(0)

  valid = (
      (row_ids >= 0)
      & (row_ids < num_rows)
      & (col_ids >= 0)
      & (col_ids < num_cols)
      & (~is_pad_skip)
  )
  r_clean = jnp.where(valid, row_ids, num_rows)
  c_clean = jnp.where(valid, col_ids, 0)

  if e_pad > 0:
    r_clean = jnp.pad(r_clean, (0, e_pad), constant_values=num_rows)
    c_clean = jnp.pad(c_clean, (0, e_pad), constant_values=0)

  use_stable = e_aligned <= block_size
  # 1. Sort edges by destination node `r_clean`. When `num_rows < 65535`, a
  # 16-bit `uint16` key halves the radix sort passes and key memory traffic.
  if num_rows < 65535:
    r_s_u16, c_s = jax.lax.sort(
        (r_clean.astype(jnp.uint16), c_clean),
        dimension=0,
        is_stable=use_stable,
        num_keys=1,
    )
    r_s = r_s_u16.astype(jnp.int32)
  else:
    r_s, c_s = jax.lax.sort(
        (r_clean, c_clean), dimension=0, is_stable=use_stable, num_keys=1
    )

  # 2. Single fused `r_pad` window: since `r_s` is sorted, all edges targeting
  # the same destination `u` are contiguous, so counting matches at offsets
  # `-8..-1` and `+1..+8` in a single padded slice pass gives the exact degree
  # for all `K in 1..8` and `>= 9` for all overflow nodes (`Category 9`) with
  # zero `jnp.roll` kernel launches and zero `bincount` scatter-adds.
  r_pad = jnp.pad(r_s, (8, 8), constant_values=-1)
  deg_capped = jnp.ones_like(r_s)
  for d in range(1, 9):
    deg_capped = deg_capped + (
        (r_pad[8 - d : 8 - d + e_aligned] == r_s).astype(jnp.int32)
        + (r_pad[8 + d : 8 + d + e_aligned] == r_s).astype(jnp.int32)
    )
  edge_cat = jnp.where(r_s >= num_rows, 10, jnp.minimum(deg_capped, 9))

  # 3. Pack `(edge_cat << 20) | r_s` into a single 32-bit key and co-sort
  # `(cat_and_r, c_s)`. Every node in Category K (1..8) has a unique key
  # `(K << 20) | u`, so all K edges of node u remain contiguous even with
  # `is_stable=False`.
  cat_and_r = jnp.bitwise_or(jnp.left_shift(edge_cat, 20), r_s)
  cat_r_sorted, c_cat = jax.lax.sort(
      (cat_and_r, c_s), dimension=0, is_stable=use_stable, num_keys=1
  )

  # 4. Binary-search category end offsets directly on `cat_r_sorted`. Category K
  # (1..10) occupies keys in `[K << 20, (K + 1) << 20)`, so its end offset is
  # the left insertion point of `(K + 1) << 20` for K = 1..10 (`2..11 << 20`).
  thresholds = jnp.arange(2, 12, dtype=jnp.int32) << 20
  b_offset_10 = jnp.searchsorted(
      cat_r_sorted, thresholds, side='left'
  ).astype(jnp.int32)
  b_offset = jnp.concatenate(
      [b_offset_10, jnp.expand_dims(kept_pad_count, 0)]
  )
  return cat_r_sorted, c_cat, b_offset


def build_tpu_spmm_schedule(
    row_ids: jax.Array,
    col_ids: jax.Array,
    num_rows: int,
    num_cols: int | None = None,
    block_size: int = _DEFAULT_COSORT_BLOCK_SIZE,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Builds a static-shape Degree-Bucketed SpMM schedule on TPU.

  Args:
    row_ids: 1D int32 array of destination node indices of shape [E].
    col_ids: 1D int32 array of source node indices of shape [E].
    num_rows: Number of output rows (destination nodes).
    num_cols: Number of input rows (source nodes). Defaults to `num_rows`.
    block_size: Number of edges streamed into SMEM per Pallas grid block.

  Returns:
    Tuple `(cat_r_sorted, c_cat, b_offset)` where `cat_r_sorted` and `c_cat`
    have fixed shape `[num_blocks * block_size]` and `b_offset` has shape
    `[11]`.
  """
  if num_cols is None:
    num_cols = num_rows
  num_rows_int = int(num_rows)
  num_cols_int = int(num_cols)
  row_ids = jnp.asarray(row_ids, dtype=jnp.int32)
  col_ids = jnp.asarray(col_ids, dtype=jnp.int32)
  e_len = row_ids.shape[0]
  if e_len == 0 or num_rows_int == 0:
    z_blk = jnp.zeros((block_size,), dtype=jnp.int32)
    b_off = jnp.zeros((11,), dtype=jnp.int32)
    return z_blk, z_blk, b_off

  if e_len < block_size:
    pad_len = block_size - e_len
    row_ids = jnp.pad(row_ids, (0, pad_len), constant_values=num_rows_int)
    col_ids = jnp.pad(col_ids, (0, pad_len), constant_values=0)

  return _build_tpu_spmm_schedule_jit(
      row_ids,
      col_ids,
      num_rows_int,
      num_cols_int,
      int(block_size),
  )


def _get_or_build_single_schedule(
    row_ids: Any,
    col_ids: Any,
    num_rows: int,
    n_in: int,
    block_size: int = _DEFAULT_COSORT_BLOCK_SIZE,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Lazily builds and caches a single-direction on-TPU CoSort schedule."""
  key = (id(row_ids), id(col_ids), int(num_rows), int(n_in), int(block_size))
  if key in _SINGLE_SCHED_CACHE:
    cached_r, cached_c, cached_sched = _SINGLE_SCHED_CACHE[key]
    if (
        cached_r is row_ids
        and cached_c is col_ids
        and getattr(cached_sched[0], '_trace', None)
        == getattr(row_ids, '_trace', None)
    ):
      _SINGLE_SCHED_CACHE.move_to_end(key)
      return cached_sched

  sched = build_tpu_spmm_schedule(
      jnp.asarray(row_ids, dtype=jnp.int32),
      jnp.asarray(col_ids, dtype=jnp.int32),
      int(num_rows),
      int(n_in),
      block_size=int(block_size),
  )
  _SINGLE_SCHED_CACHE[key] = (row_ids, col_ids, sched)
  if len(_SINGLE_SCHED_CACHE) > _MAX_SCHEDULE_CACHE_SIZE:
    _SINGLE_SCHED_CACHE.popitem(last=False)
  return sched


def prepare_spmm_schedule(
    row_ids: Any,
    col_ids: Any,
    num_rows: int,
    n_in: int | None = None,
    *,
    block_size: int = _DEFAULT_COSORT_BLOCK_SIZE,
    unroll: int = _DEFAULT_COSORT_UNROLL,
) -> tuple[
    tuple[jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array, jax.Array],
]:
  """Builds and caches forward and backward on-TPU Co-Sort schedules."""
  del unroll
  if n_in is None:
    n_in = num_rows
  if (
      block_size == _DEFAULT_BUCKET_BLOCK_SIZE
      and jax.default_backend() == 'tpu'
  ):
    eff_block_size = _DEFAULT_COSORT_BLOCK_SIZE
  else:
    eff_block_size = block_size
  key = (id(row_ids), id(col_ids), int(num_rows), int(n_in), eff_block_size)
  if key in _PAIR_SCHED_CACHE:
    cached_r, cached_c, cached_pair = _PAIR_SCHED_CACHE[key]
    if (
        cached_r is row_ids
        and cached_c is col_ids
        and getattr(cached_pair[0][0], '_trace', None)
        == getattr(row_ids, '_trace', None)
    ):
      _PAIR_SCHED_CACHE.move_to_end(key)
      return cached_pair

  sched_fwd = _get_or_build_single_schedule(
      row_ids, col_ids, int(num_rows), int(n_in), block_size=eff_block_size
  )
  sched_bwd = _get_or_build_single_schedule(
      col_ids, row_ids, int(n_in), int(num_rows), block_size=eff_block_size
  )
  pair = (sched_fwd, sched_bwd)
  _PAIR_SCHED_CACHE[key] = (row_ids, col_ids, pair)
  if len(_PAIR_SCHED_CACHE) > _MAX_SCHEDULE_CACHE_SIZE:
    _PAIR_SCHED_CACHE.popitem(last=False)
  return pair


@functools.cache
def _make_cosort_spmm_kernel(
    num_rows: int,
    n_in: int,
    m_tile: int,
    l_tile: int,
    num_blocks: int,
    block_size: int = _DEFAULT_COSORT_BLOCK_SIZE,
    unroll: int = _DEFAULT_COSORT_UNROLL,
    use_fp32_acc: bool = False,
    use_vreg_fp32_acc: bool = False,
    diag_weight: float = 0.0,
    has_col_grid: bool = False,
    emulate_seq_pad: bool = True,
):
  """Builds the Degree-Bucketed Pallas Mosaic TPU kernel."""
  zero_chunk = min(4096, num_rows)
  num_zero_chunks = num_rows // zero_chunk
  zero_rem = num_rows % zero_chunk
  has_diag = diag_weight != 0.0 and num_rows == n_in

  def _kernel(
      b_offset_ref, cat_r_ref, c_cat_ref, x_ref, out_ref, *scratch_rest
  ):
    if use_vreg_fp32_acc:
      carry_d_smem, carry_vmem = scratch_rest
      acc_ref = out_ref
      acc_dtype = jnp.float32
    else:
      carry_d_smem = None
      carry_vmem = None
      acc_ref = scratch_rest[0] if use_fp32_acc else out_ref
      acc_dtype = acc_ref.dtype

    blk = pl.program_id(1) if has_col_grid else pl.program_id(0)
    is_last_blk = blk == (num_blocks - 1)
    elem_start = blk * block_size
    elem_end = elem_start + block_size
    dw = jnp.asarray(diag_weight, dtype=acc_dtype)

    @pl.when(blk == 0)
    def _zero():
      if (
          use_vreg_fp32_acc
          and carry_d_smem is not None
          and carry_vmem is not None
      ):
        carry_d_smem[0] = jnp.int32(-1)
        carry_vmem[pl.ds(0, 1)] = jnp.zeros(
            (1, m_tile, l_tile), dtype=jnp.float32
        )
      if use_vreg_fp32_acc and has_diag:
        if num_zero_chunks > 0:

          def _z_diag(i, _):
            s = pl.ds(i * zero_chunk, zero_chunk)
            out_ref[s] = (x_ref[s].astype(jnp.float32) * dw).astype(
                out_ref.dtype
            )
            return None

          jax.lax.fori_loop(0, num_zero_chunks, _z_diag, None)
        if zero_rem > 0:
          s_rem = pl.ds(num_zero_chunks * zero_chunk, zero_rem)
          out_ref[s_rem] = (x_ref[s_rem].astype(jnp.float32) * dw).astype(
              out_ref.dtype
          )
      else:
        if num_zero_chunks > 0:
          z = jnp.zeros((zero_chunk, m_tile, l_tile), dtype=acc_ref.dtype)

          def _zb(i, _):
            acc_ref[pl.ds(i * zero_chunk, zero_chunk)] = z
            return None

          jax.lax.fori_loop(0, num_zero_chunks, _zb, None)
        if zero_rem > 0:
          acc_ref[pl.ds(num_zero_chunks * zero_chunk, zero_rem)] = jnp.zeros(
              (zero_rem, m_tile, l_tile), dtype=acc_ref.dtype
          )

    def _load(c_ref, idx):
      v = x_ref[pl.ds(c_ref[idx], 1)]
      return (
          v.astype(acc_dtype)
          if (use_fp32_acc or use_vreg_fp32_acc)
          else v
      )

    def _finish_node_val(d, val_f32):
      if use_vreg_fp32_acc:
        if has_diag:
          val_f32 = val_f32 + x_ref[pl.ds(d, 1)].astype(jnp.float32) * dw
        return val_f32.astype(out_ref.dtype)
      return val_f32

    if unroll == 1:
      hi_loc = jnp.minimum(
          block_size, jnp.maximum(0, b_offset_ref[8] - elem_start)
      )

      def _interp_body(idx, _):
        d = jnp.bitwise_and(cat_r_ref[idx], 0xFFFFF)
        acc_ref[pl.ds(d, 1)] = acc_ref[pl.ds(d, 1)] + _load(c_cat_ref, idx)
        return None

      jax.lax.fori_loop(0, hi_loc, _interp_body, None)
    else:

      def _emit_category(k_edges: int):
        u_k = max(4, min(unroll, 16 // k_edges))
        stride = u_k * k_edges
        cat_start = 0 if k_edges == 1 else b_offset_ref[k_edges - 2]
        cat_end = b_offset_ref[k_edges - 1]
        lo_g = jnp.maximum(elem_start, cat_start)
        hi_g = jnp.minimum(elem_end, cat_end)

        @pl.when(hi_g > lo_g)
        def _run_k():
          if num_blocks > 1 and k_edges > 1:
            g_first = jnp.minimum(
                hi_g,
                cat_start
                + ((lo_g - cat_start + k_edges - 1) // k_edges) * k_edges,
            )
            g_last = jnp.maximum(
                g_first, cat_start + ((hi_g - cat_start) // k_edges) * k_edges
            )
            lo_loc = lo_g - elem_start
            first_loc = g_first - elem_start
            last_loc = g_last - elem_start
            hi_loc = hi_g - elem_start

            if use_vreg_fp32_acc and carry_vmem is not None:

              @pl.when(first_loc > lo_loc)
              def _prefix_vreg():
                d_pre = jnp.bitwise_and(cat_r_ref[lo_loc], 0xFFFFF)
                acc_pre = carry_vmem[pl.ds(0, 1)]

                def _pre_loop(idx, acc_in):
                  return acc_in + _load(c_cat_ref, idx)

                acc_pre = jax.lax.fori_loop(
                    lo_loc, first_loc, _pre_loop, acc_pre
                )
                out_ref[pl.ds(d_pre, 1)] = _finish_node_val(d_pre, acc_pre)

            else:

              def _prefix_body(idx, _):
                d = jnp.bitwise_and(cat_r_ref[idx], 0xFFFFF)
                acc_ref[pl.ds(d, 1)] = acc_ref[pl.ds(d, 1)] + _load(
                    c_cat_ref, idx
                )
                return None

              jax.lax.fori_loop(lo_loc, first_loc, _prefix_body, None)
          else:
            first_loc = lo_g - elem_start
            last_loc = hi_g - elem_start
            hi_loc = last_loc

          num_tuples = (last_loc - first_loc) // k_edges
          n_steps = num_tuples // u_k
          rem_tuples_start = n_steps * u_k

          def _unrolled_body(step, _):
            base = first_loc + step * stride
            dsts = [
                jnp.bitwise_and(
                    cat_r_ref[base if u == 0 else base + (u * k_edges)], 0xFFFFF
                )
                for u in range(u_k)
            ]
            vals = [
                _load(c_cat_ref, base if u == 0 else base + (u * k_edges))
                for u in range(u_k)
            ]
            for j in range(1, k_edges):
              vj = [
                  _load(c_cat_ref, base + (u * k_edges + j))
                  for u in range(u_k)
              ]
              vals = [vals[u] + vj[u] for u in range(u_k)]
            for u in range(u_k):
              acc_ref[pl.ds(dsts[u], 1)] = _finish_node_val(dsts[u], vals[u])
            return None

          jax.lax.fori_loop(0, n_steps, _unrolled_body, None)

          def _rem_body(m, _):
            pos = first_loc + m * k_edges
            d = jnp.bitwise_and(cat_r_ref[pos], 0xFFFFF)
            v_sum = _load(c_cat_ref, pos)
            for j in range(1, k_edges):
              v_sum = v_sum + _load(c_cat_ref, pos + j)
            acc_ref[pl.ds(d, 1)] = _finish_node_val(d, v_sum)
            return None

          jax.lax.fori_loop(rem_tuples_start, num_tuples, _rem_body, None)

          if num_blocks > 1 and k_edges > 1:
            if use_vreg_fp32_acc and carry_vmem is not None:

              @pl.when(hi_loc > last_loc)
              def _suffix_vreg():
                acc_suf = _load(c_cat_ref, last_loc)

                def _suf_loop(idx, acc_in):
                  return acc_in + _load(c_cat_ref, idx)

                acc_suf = jax.lax.fori_loop(
                    last_loc + 1, hi_loc, _suf_loop, acc_suf
                )
                carry_vmem[pl.ds(0, 1)] = acc_suf

            else:

              def _suffix_body(idx, _):
                d = jnp.bitwise_and(cat_r_ref[idx], 0xFFFFF)
                acc_ref[pl.ds(d, 1)] = acc_ref[pl.ds(d, 1)] + _load(
                    c_cat_ref, idx
                )
                return None

              jax.lax.fori_loop(last_loc, hi_loc, _suffix_body, None)

      for k_val in range(1, 9):
        _emit_category(k_val)

      # Category 9: Overflow nodes (`deg >= 9`) in
      # `[b_offset_ref[7], b_offset_ref[8])`.
      cat9_start = b_offset_ref[7]
      cat9_end = b_offset_ref[8]
      lo9_g = jnp.maximum(elem_start, cat9_start)
      hi9_g = jnp.minimum(elem_end, cat9_end)

      @pl.when(hi9_g > lo9_g)
      def _run_ov():
        lo9_loc = lo9_g - elem_start
        hi9_loc = hi9_g - elem_start
        cnt9 = hi9_loc - lo9_loc

        if (
            use_vreg_fp32_acc
            and carry_d_smem is not None
            and carry_vmem is not None
        ):
          z = jnp.zeros((1, m_tile, l_tile), dtype=jnp.float32)
          is_first_ov = lo9_g == cat9_start
          first_d = jnp.bitwise_and(cat_r_ref[lo9_loc], 0xFFFFF)
          prev_d0 = jnp.where(is_first_ov, first_d, carry_d_smem[0])
          o0 = jnp.where(is_first_ov, z, carry_vmem[pl.ds(0, 1)])

          if num_blocks > 1:
            n_steps9 = cnt9 // 8
            rem9_start = lo9_loc + n_steps9 * 8

            def _step9_vreg(step, state):
              prev_d, o = state
              pos = lo9_loc + step * 8
              ds = [
                  jnp.bitwise_and(
                      cat_r_ref[pos if u == 0 else pos + u], 0xFFFFF
                  )
                  for u in range(8)
              ]
              vs = [
                  _load(c_cat_ref, pos if u == 0 else pos + u) for u in range(8)
              ]
              m = [ds[u] == prev_d for u in range(8)]
              vp0 = jnp.where(m[0], vs[0], z) + jnp.where(m[1], vs[1], z)
              vp1 = jnp.where(m[2], vs[2], z) + jnp.where(m[3], vs[3], z)
              vp2 = jnp.where(m[4], vs[4], z) + jnp.where(m[5], vs[5], z)
              vp3 = jnp.where(m[6], vs[6], z) + jnp.where(m[7], vs[7], z)
              vp = (vp0 + vp1) + (vp2 + vp3)

              vn0 = jnp.where(m[0], z, vs[0]) + jnp.where(m[1], z, vs[1])
              vn1 = jnp.where(m[2], z, vs[2]) + jnp.where(m[3], z, vs[3])
              vn2 = jnp.where(m[4], z, vs[4]) + jnp.where(m[5], z, vs[5])
              vn3 = jnp.where(m[6], z, vs[6]) + jnp.where(m[7], z, vs[7])
              vn = (vn0 + vn1) + (vn2 + vn3)

              o_prev = o + vp
              out_ref[pl.ds(prev_d, 1)] = _finish_node_val(prev_d, o_prev)
              o_next = jnp.where(m[7], o_prev, vn)
              return (ds[7], o_next)

            prev_d1, o1 = jax.lax.fori_loop(
                0, n_steps9, _step9_vreg, (prev_d0, o0)
            )
          else:
            n_steps9 = cnt9 // 4
            rem9_start = lo9_loc + n_steps9 * 4

            def _step9_vreg_seq(step, state):
              prev_d, o = state
              pos = lo9_loc + step * 4
              ds = [
                  jnp.bitwise_and(
                      cat_r_ref[pos if u == 0 else pos + u], 0xFFFFF
                  )
                  for u in range(4)
              ]
              vs = [
                  _load(c_cat_ref, pos if u == 0 else pos + u) for u in range(4)
              ]
              for u in range(4):
                d_u = ds[u]
                v_u = vs[u]
                o_prev = jnp.where(d_u == prev_d, o + v_u, o)
                out_ref[pl.ds(prev_d, 1)] = _finish_node_val(prev_d, o_prev)
                o = jnp.where(d_u == prev_d, o_prev, v_u)
                prev_d = d_u
              return (prev_d, o)

            prev_d1, o1 = jax.lax.fori_loop(
                0, n_steps9, _step9_vreg_seq, (prev_d0, o0)
            )

          def _rem9_vreg(idx, state):
            prev_d, o = state
            d = jnp.bitwise_and(cat_r_ref[idx], 0xFFFFF)
            v = _load(c_cat_ref, idx)
            o_prev = jnp.where(d == prev_d, o + v, o)
            out_ref[pl.ds(prev_d, 1)] = _finish_node_val(prev_d, o_prev)
            o_next = jnp.where(d == prev_d, o_prev, v)
            return (d, o_next)

          prev_d2, o2 = jax.lax.fori_loop(
              rem9_start, hi9_loc, _rem9_vreg, (prev_d1, o1)
          )
          out_ref[pl.ds(prev_d2, 1)] = _finish_node_val(prev_d2, o2)
          carry_d_smem[0] = prev_d2
          carry_vmem[pl.ds(0, 1)] = o2
        else:
          n_steps9 = cnt9 // 4
          rem9_start = lo9_loc + n_steps9 * 4

          def _step9(step, _):
            pos = lo9_loc + step * 4
            ds = [
                jnp.bitwise_and(
                    cat_r_ref[pos if u == 0 else pos + u], 0xFFFFF
                )
                for u in range(4)
            ]
            vs = [
                _load(c_cat_ref, pos if u == 0 else pos + u) for u in range(4)
            ]
            us = [acc_ref[pl.ds(ds[u], 1)] for u in range(4)]
            o = us[0] + vs[0]
            acc_ref[pl.ds(ds[0], 1)] = o
            for u in range(1, 4):
              o = jnp.where(ds[u] == ds[u - 1], o, us[u]) + vs[u]
              acc_ref[pl.ds(ds[u], 1)] = o
            return None

          jax.lax.fori_loop(0, n_steps9, _step9, None)

          def _rem9(idx, _):
            d = jnp.bitwise_and(cat_r_ref[idx], 0xFFFFF)
            acc_ref[pl.ds(d, 1)] = acc_ref[pl.ds(d, 1)] + _load(c_cat_ref, idx)
            return None

          jax.lax.fori_loop(rem9_start, hi9_loc, _rem9, None)

    # Apply collapsed FixedSizePadder padding self-loops on the final block.
    @pl.when(is_last_blk)
    def _finalize_pad_and_out():
      pad_cnt = b_offset_ref[10]

      @pl.when(pad_cnt > 0)
      def _apply_pad_self():
        pad_dst = num_rows - 1
        pad_src = n_in - 1
        vp = x_ref[pl.ds(pad_src, 1)]
        if use_fp32_acc or use_vreg_fp32_acc:
          vp = vp.astype(jnp.float32)
        if use_vreg_fp32_acc:
          up = jnp.zeros((1, m_tile, l_tile), dtype=jnp.float32)
        else:
          up = acc_ref[pl.ds(pad_dst, 1)]
        def _seq_pad(u_in):
          chunks64 = jax.lax.div(pad_cnt, 64)
          rem64 = pad_cnt - chunks64 * 64

          def _p64(_, u_acc):
            for _ in range(64):
              u_acc = u_acc + vp
            return u_acc

          u_mid = jax.lax.fori_loop(0, chunks64, _p64, u_in)

          def _p1(_, u_acc):
            return u_acc + vp

          return jax.lax.fori_loop(0, rem64, _p1, u_mid)

        if emulate_seq_pad:
          up = _seq_pad(up)
        else:
          up = jax.lax.cond(
              pad_cnt < 4096,
              _seq_pad,
              lambda u_in: u_in + vp * pad_cnt.astype(vp.dtype),
              up,
          )
        if use_vreg_fp32_acc:
          out_ref[pl.ds(pad_dst, 1)] = _finish_node_val(pad_dst, up)
        else:
          acc_ref[pl.ds(pad_dst, 1)] = up

      if use_fp32_acc and not use_vreg_fp32_acc:
        if num_zero_chunks > 0:

          def _cast_body(i, _):
            s = pl.ds(i * zero_chunk, zero_chunk)
            val = acc_ref[s]
            if has_diag:
              val = val + x_ref[s].astype(acc_ref.dtype) * dw
            out_ref[s] = val.astype(out_ref.dtype)
            return None

          jax.lax.fori_loop(0, num_zero_chunks, _cast_body, None)
        if zero_rem > 0:
          s_rem = pl.ds(num_zero_chunks * zero_chunk, zero_rem)
          val_rem = acc_ref[s_rem]
          if has_diag:
            val_rem = val_rem + x_ref[s_rem].astype(acc_ref.dtype) * dw
          out_ref[s_rem] = val_rem.astype(out_ref.dtype)
      elif has_diag and not use_vreg_fp32_acc:
        if num_zero_chunks > 0:

          def _diag_body(i, _):
            s = pl.ds(i * zero_chunk, zero_chunk)
            acc_ref[s] = acc_ref[s] + x_ref[s] * dw
            return None

          jax.lax.fori_loop(0, num_zero_chunks, _diag_body, None)
        if zero_rem > 0:
          s_rem = pl.ds(num_zero_chunks * zero_chunk, zero_rem)
          acc_ref[s_rem] = acc_ref[s_rem] + x_ref[s_rem] * dw

  return _kernel


def _run_cosort_spmm_2d(
    x_2d: jax.Array,
    schedule: tuple[jax.Array, jax.Array, jax.Array],
    num_rows: int,
    *,
    block_size: int = _DEFAULT_COSORT_BLOCK_SIZE,
    unroll: int = _DEFAULT_COSORT_UNROLL,
    interpret: bool = False,
    diag_weight: float = 0.0,
    device_profile: TPUDeviceProfile | str | None = None,
) -> jax.Array:
  """Executes the Single-Sort Degree-Bucketed Pallas SpMM kernel on a 2D matrix."""
  n_in, feat_dim = x_2d.shape
  if n_in == 0 or num_rows == 0 or feat_dim == 0:
    y_zero = jnp.zeros((num_rows, feat_dim), dtype=x_2d.dtype)
    if diag_weight != 0.0 and num_rows == n_in:
      return y_zero + x_2d * jnp.asarray(diag_weight, dtype=x_2d.dtype)
    return y_zero

  profile = get_tpu_device_profile(device_profile)
  max_working_set = _effective_max_working_set_bytes(profile)
  emulate_seq_pad = profile.generation < 7

  itemsize = jnp.dtype(x_2d.dtype).itemsize
  need_fp32_acc = itemsize < 4
  cat_r_sorted, c_cat, b_offset = schedule
  num_blocks = cat_r_sorted.shape[0] // block_size
  b_offset_pad = jnp.pad(b_offset, (0, 5), constant_values=0)

  m_tile, l_tile, d_pad = pick_3d_tile_shape(feat_dim, x_2d.dtype)
  if d_pad > feat_dim:
    x_padded = jnp.pad(x_2d, ((0, 0), (0, d_pad - feat_dim)))
  else:
    x_padded = x_2d

  # When `not interpret`, `_make_cosort_spmm_kernel` accumulates in `float32`
  # VREGs + a 1-node `carry_vmem` buffer (`512B–1KiB`), avoiding both the
  # `4 * num_rows * D` VMEM scratch buffer on `bfloat16` and destination VMEM
  # Read-After-Write stalls in Category 9 / Prefix / Suffix on both `float32`
  # and `bfloat16`.
  vreg_bytes_per_col = (n_in + num_rows) * itemsize
  full_bytes_per_col = n_in * itemsize + num_rows * (
      itemsize + (4 if need_fp32_acc else 0)
  )
  use_vreg_fp32_acc = not interpret
  use_fp32_acc = need_fp32_acc and not use_vreg_fp32_acc
  bytes_per_col = (
      vreg_bytes_per_col if use_vreg_fp32_acc else full_bytes_per_col
  )
  working_set_bytes = bytes_per_col * d_pad

  smem_space = None if interpret else pltpu.SMEM
  vmem_buf = None if interpret else pl.Buffered(buffer_count=1)

  def _call_tile(
      x_3d_tile: jax.Array, m_t: int, l_t: int, g_col: int = 1
  ) -> jax.Array:
    has_col_grid = g_col > 1
    kernel_fn = _make_cosort_spmm_kernel(
        num_rows=num_rows,
        n_in=n_in,
        m_tile=m_t,
        l_tile=l_t,
        num_blocks=num_blocks,
        block_size=block_size,
        unroll=unroll,
        use_fp32_acc=use_fp32_acc,
        use_vreg_fp32_acc=use_vreg_fp32_acc,
        diag_weight=diag_weight,
        has_col_grid=has_col_grid,
        emulate_seq_pad=emulate_seq_pad,
    )
    if use_vreg_fp32_acc:
      scratch: Any = [
          pltpu.SMEM((1,), jnp.int32),
          pltpu.VMEM((1, m_t, l_t), jnp.float32),
      ]
    elif use_fp32_acc:
      scratch = [pltpu.VMEM((num_rows, m_t, l_t), jnp.float32)]
    else:
      scratch = ()

    if has_col_grid:
      b_spec = pl.BlockSpec((16,), lambda c, b: (0,), memory_space=smem_space)
      e_spec = pl.BlockSpec(
          (block_size,), lambda c, b: (b,), memory_space=smem_space
      )
      if interpret:
        compiler_params = None
      else:
        compiler_params = pltpu.CompilerParams(
            dimension_semantics=('parallel', 'arbitrary'),
            vmem_limit_bytes=profile.vmem_limit_bytes,
            disable_bounds_checks=True,
        )
      return pl.pallas_call(
          kernel_fn,
          out_shape=jax.ShapeDtypeStruct(
              (num_rows, m_t, g_col * l_t), x_2d.dtype
          ),
          grid=(g_col, num_blocks),
          in_specs=[
              b_spec,
              e_spec,
              e_spec,
              pl.BlockSpec(
                  (n_in, m_t, l_t),
                  lambda c, b: (0, 0, c),
                  pipeline_mode=vmem_buf,
              ),
          ],
          out_specs=pl.BlockSpec(
              (num_rows, m_t, l_t),
              lambda c, b: (0, 0, c),
              pipeline_mode=vmem_buf,
          ),
          scratch_shapes=scratch,
          compiler_params=compiler_params,
          interpret=interpret,
          name='pallas_cosort_spmm_kernel',
      )(b_offset_pad, cat_r_sorted, c_cat, x_3d_tile)

    b_spec = pl.BlockSpec((16,), lambda b: (0,), memory_space=smem_space)
    e_spec = pl.BlockSpec(
        (block_size,), lambda b: (b,), memory_space=smem_space
    )
    if interpret:
      compiler_params = None
    else:
      compiler_params = pltpu.CompilerParams(
          dimension_semantics=('arbitrary',),
          vmem_limit_bytes=profile.vmem_limit_bytes,
          disable_bounds_checks=True,
      )
    return pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct((num_rows, m_t, l_t), x_2d.dtype),
        grid=(num_blocks,),
        in_specs=[
            b_spec,
            e_spec,
            e_spec,
            pl.BlockSpec(
                (n_in, m_t, l_t), lambda b: (0, 0, 0), pipeline_mode=vmem_buf
            ),
        ],
        out_specs=pl.BlockSpec(
            (num_rows, m_t, l_t), lambda b: (0, 0, 0), pipeline_mode=vmem_buf
        ),
        scratch_shapes=scratch,
        compiler_params=compiler_params,
        interpret=interpret,
        name='pallas_cosort_spmm_kernel',
    )(b_offset_pad, cat_r_sorted, c_cat, x_3d_tile)

  if working_set_bytes <= max_working_set:
    x_3d = x_padded.reshape(n_in, m_tile, l_tile)
    y_3d = _call_tile(x_3d, m_tile, l_tile)
    y_2d = y_3d.reshape(num_rows, d_pad)
  else:
    packing = max(1, 4 // itemsize)
    max_chunk_dim = max(
        packing * 128, max_working_set // max(1, bytes_per_col)
    )
    chunk_dim = 256 if max_chunk_dim >= 256 else 128
    chunk_m, chunk_l, chunk_dim = pick_3d_tile_shape(chunk_dim, x_2d.dtype)
    if d_pad % chunk_dim != 0:
      d_pad = ((d_pad + chunk_dim - 1) // chunk_dim) * chunk_dim
      x_padded = jnp.pad(x_2d, ((0, 0), (0, d_pad - feat_dim)))
    g_col = d_pad // chunk_dim
    x_3d = x_padded.reshape(n_in, chunk_m, g_col * chunk_l)
    y_3d = _call_tile(x_3d, chunk_m, chunk_l, g_col=g_col)
    y_2d = y_3d.reshape(num_rows, d_pad)

  if d_pad > feat_dim:
    y_2d = y_2d[:, :feat_dim]
  return y_2d


@functools.cache
def _make_spmm_kernel(
    num_rows: int,
    n_in: int,
    m_tile: int,
    l_tile: int,
    edge_block_size: int,
    num_edge_blocks: int,
    last_block_edges: int,
    unroll: int,
    weighted: bool = False,
    use_fp32_acc: bool = False,
    diag_weight: float = 0.0,
    has_col_grid: bool = False,
):
  """Builds direct-edge Pallas Mosaic TPU kernel with store-to-load forwarding."""
  full_loop_steps = edge_block_size // unroll
  last_loop_steps = last_block_edges // unroll
  last_rem_edges = last_block_edges % unroll

  zero_chunk = min(_ZERO_CHUNK_NODES, num_rows)
  num_zero_chunks = num_rows // zero_chunk
  zero_rem = num_rows % zero_chunk
  has_diag = diag_weight != 0.0 and num_rows == n_in

  def _kernel(row_ref, col_ref, *rest):
    if weighted:
      if use_fp32_acc:
        val_ref, x_ref, y_ref, acc_ref = rest
      else:
        val_ref, x_ref, y_ref = rest
        acc_ref = y_ref
    else:
      if use_fp32_acc:
        x_ref, y_ref, acc_ref = rest
      else:
        x_ref, y_ref = rest
        acc_ref = y_ref
      val_ref = None

    prog_id = pl.program_id(axis=1 if has_col_grid else 0)
    is_last_blk = prog_id == (num_edge_blocks - 1)

    # Initialize accumulator (acc_ref) to 0 in VMEM on the first edge block.
    @pl.when(prog_id == 0)
    def _zero_acc():
      if num_zero_chunks > 0:
        z_chunk = jnp.zeros((zero_chunk, m_tile, l_tile), dtype=acc_ref.dtype)

        def _zero_chunk_body(i, _):
          acc_ref[pl.ds(i * zero_chunk, zero_chunk)] = z_chunk
          return None

        jax.lax.fori_loop(0, num_zero_chunks, _zero_chunk_body, None)
      if zero_rem > 0:
        acc_ref[pl.ds(num_zero_chunks * zero_chunk, zero_rem)] = jnp.zeros(
            (zero_rem, m_tile, l_tile), dtype=acc_ref.dtype
        )

    def _edge_loop_body(step, _):
      base = step * unroll
      if unroll == 1:
        src0 = col_ref[base]
        dst0 = row_ref[base]
        v0 = x_ref[pl.ds(src0, 1)]
        if weighted and val_ref is not None:
          w0 = val_ref[base].astype(x_ref.dtype)
          v0 = v0 * w0
        if use_fp32_acc:
          v0 = v0.astype(acc_ref.dtype)
        u0 = acc_ref[pl.ds(dst0, 1)]
        acc_ref[pl.ds(dst0, 1)] = u0 + v0
        return None

      srcs = [col_ref[base if k == 0 else base + k] for k in range(unroll)]
      dsts = [row_ref[base if k == 0 else base + k] for k in range(unroll)]

      vs = [x_ref[pl.ds(srcs[k], 1)] for k in range(unroll)]
      if weighted and val_ref is not None:
        ws = [
            val_ref[base if k == 0 else base + k].astype(x_ref.dtype)
            for k in range(unroll)
        ]
        vs = [vs[k] * ws[k] for k in range(unroll)]
      if use_fp32_acc:
        vs = [vs[k].astype(acc_ref.dtype) for k in range(unroll)]

      us = [acc_ref[pl.ds(dsts[k], 1)] for k in range(unroll)]

      # Store-to-load forwarding in registers across the unroll window:
      # if dsts[k] == dsts[j] for an earlier lane j < k, forward ys[j].
      ys = [us[0] + vs[0]]
      for k in range(1, unroll):
        u_k = us[k]
        for j in range(k):
          u_k = jnp.where(dsts[k] == dsts[j], ys[j], u_k)
        ys.append(u_k + vs[k])

      for k in range(unroll):
        acc_ref[pl.ds(dsts[k], 1)] = ys[k]
      return None

    if last_block_edges == edge_block_size:
      jax.lax.fori_loop(0, full_loop_steps, _edge_loop_body, None)
    else:
      n_steps = jnp.where(is_last_blk, last_loop_steps, full_loop_steps)
      jax.lax.fori_loop(0, n_steps, _edge_loop_body, None)
      if last_rem_edges > 0:

        @pl.when(is_last_blk)
        def _tail_edges():
          base_tail = last_loop_steps * unroll
          for r in range(last_rem_edges):
            src_r = col_ref[base_tail + r]
            dst_r = row_ref[base_tail + r]
            vr = x_ref[pl.ds(src_r, 1)]
            if weighted and val_ref is not None:
              vr = vr * val_ref[base_tail + r].astype(x_ref.dtype)
            if use_fp32_acc:
              vr = vr.astype(acc_ref.dtype)
            acc_ref[pl.ds(dst_r, 1)] = acc_ref[pl.ds(dst_r, 1)] + vr

    if use_fp32_acc:

      @pl.when(is_last_blk)
      def _finalize_y():
        dw = jnp.asarray(diag_weight, dtype=acc_ref.dtype)
        if num_zero_chunks > 0:

          def _cast_chunk_body(i, _):
            idx = pl.ds(i * zero_chunk, zero_chunk)
            val = acc_ref[idx]
            if has_diag:
              val = val + x_ref[idx].astype(acc_ref.dtype) * dw
            y_ref[idx] = val.astype(y_ref.dtype)
            return None

          jax.lax.fori_loop(0, num_zero_chunks, _cast_chunk_body, None)
        if zero_rem > 0:
          rem_idx = pl.ds(num_zero_chunks * zero_chunk, zero_rem)
          val_rem = acc_ref[rem_idx]
          if has_diag:
            val_rem = val_rem + x_ref[rem_idx].astype(acc_ref.dtype) * dw
          y_ref[rem_idx] = val_rem.astype(y_ref.dtype)
    elif has_diag:

      @pl.when(is_last_blk)
      def _finalize_diag():
        dw = jnp.asarray(diag_weight, dtype=acc_ref.dtype)
        if num_zero_chunks > 0:

          def _diag_chunk_body(i, _):
            idx = pl.ds(i * zero_chunk, zero_chunk)
            acc_ref[idx] = acc_ref[idx] + x_ref[idx] * dw
            return None

          jax.lax.fori_loop(0, num_zero_chunks, _diag_chunk_body, None)
        if zero_rem > 0:
          rem_idx = pl.ds(num_zero_chunks * zero_chunk, zero_rem)
          acc_ref[rem_idx] = acc_ref[rem_idx] + x_ref[rem_idx] * dw

  return _kernel


def _cpu_chunked_spmm(
    x_2d: jax.Array,
    row_ids: jax.Array,
    col_ids: jax.Array,
    num_rows: int,
    values: jax.Array | None = None,
    chunk_size: int = 4096,
    diag_weight: float = 0.0,
) -> jax.Array:
  """O(N)-memory chunked SpMM fallback for CPU with fp32 accumulation."""
  num_edges = row_ids.shape[0]
  feat_dim = x_2d.shape[1]
  acc_dtype = (
      jnp.float32 if jnp.dtype(x_2d.dtype).itemsize < 4 else x_2d.dtype
  )
  y = jnp.zeros((num_rows, feat_dim), dtype=acc_dtype)
  if num_edges > 0:
    num_full_chunks = num_edges // chunk_size
    if num_full_chunks > 0:

      def _chunk_body(i, acc):
        start = i * chunk_size
        r_chunk = jax.lax.dynamic_slice_in_dim(row_ids, start, chunk_size)
        c_chunk = jax.lax.dynamic_slice_in_dim(col_ids, start, chunk_size)
        gathered = x_2d[c_chunk]
        if values is not None:
          v_chunk = jax.lax.dynamic_slice_in_dim(values, start, chunk_size)
          gathered = gathered * v_chunk[:, None].astype(x_2d.dtype)
        return acc.at[r_chunk].add(gathered.astype(acc_dtype))

      y = jax.lax.fori_loop(0, num_full_chunks, _chunk_body, y)

    rem = num_edges % chunk_size
    if rem > 0:
      start = num_full_chunks * chunk_size
      r_rem = row_ids[start:]
      c_rem = col_ids[start:]
      gathered = x_2d[c_rem]
      if values is not None:
        v_rem = values[start:]
        gathered = gathered * v_rem[:, None].astype(x_2d.dtype)
      y = y.at[r_rem].add(gathered.astype(acc_dtype))

  if diag_weight != 0.0 and num_rows == x_2d.shape[0]:
    y = y + x_2d.astype(acc_dtype) * jnp.asarray(diag_weight, dtype=acc_dtype)
  return y.astype(x_2d.dtype)


def _run_single_3d_tile(
    x_3d: jax.Array,
    row_ids_aligned: jax.Array,
    col_ids_aligned: jax.Array,
    values_aligned: jax.Array | None,
    num_rows: int,
    m_tile: int,
    l_tile: int,
    edge_block_size: int,
    num_edge_blocks: int,
    last_block_edges: int,
    unroll: int,
    interpret: bool,
    diag_weight: float = 0.0,
    vmem_limit_bytes: int = _DEFAULT_VMEM_LIMIT_BYTES,
    g_col: int = 1,
) -> jax.Array:
  """Runs 1D or 2D-grid Pallas SpMM on 3D feature tiles (n_in, m_tile, g_col * l_tile)."""
  n_in = x_3d.shape[0]
  weighted = values_aligned is not None
  use_fp32_acc = jnp.dtype(x_3d.dtype).itemsize < 4
  has_col_grid = g_col > 1
  kernel_fn = _make_spmm_kernel(
      num_rows=num_rows,
      n_in=n_in,
      m_tile=m_tile,
      l_tile=l_tile,
      edge_block_size=edge_block_size,
      num_edge_blocks=num_edge_blocks,
      last_block_edges=last_block_edges,
      unroll=unroll,
      weighted=weighted,
      use_fp32_acc=use_fp32_acc,
      diag_weight=diag_weight,
      has_col_grid=has_col_grid,
  )

  smem_space = None if interpret else pltpu.SMEM
  vmem_single_buffer = None if interpret else pl.Buffered(buffer_count=1)
  if interpret:
    compiler_params = None
  else:
    dim_semantics = (
        ('parallel', 'arbitrary') if has_col_grid else ('arbitrary',)
    )
    compiler_params = pltpu.CompilerParams(
        dimension_semantics=dim_semantics,
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )

  out_shape = jax.ShapeDtypeStruct(
      (num_rows, m_tile, g_col * l_tile), x_3d.dtype
  )
  if has_col_grid:
    grid = (g_col, num_edge_blocks)
    row_spec = pl.BlockSpec(
        (edge_block_size,), lambda c, e_blk: (e_blk,), memory_space=smem_space
    )
    col_spec = pl.BlockSpec(
        (edge_block_size,), lambda c, e_blk: (e_blk,), memory_space=smem_space
    )
    x_spec = pl.BlockSpec(
        (n_in, m_tile, l_tile),
        lambda c, e_blk: (0, 0, c),
        pipeline_mode=vmem_single_buffer,
    )
    y_spec = pl.BlockSpec(
        (num_rows, m_tile, l_tile),
        lambda c, e_blk: (0, 0, c),
        pipeline_mode=vmem_single_buffer,
    )
    val_spec_fn = lambda c, e_blk: (e_blk,)
  else:
    grid = (num_edge_blocks,)
    row_spec = pl.BlockSpec(
        (edge_block_size,), lambda e_blk: (e_blk,), memory_space=smem_space
    )
    col_spec = pl.BlockSpec(
        (edge_block_size,), lambda e_blk: (e_blk,), memory_space=smem_space
    )
    x_spec = pl.BlockSpec(
        (n_in, m_tile, l_tile),
        lambda e_blk: (0, 0, 0),
        pipeline_mode=vmem_single_buffer,
    )
    y_spec = pl.BlockSpec(
        (num_rows, m_tile, l_tile),
        lambda e_blk: (0, 0, 0),
        pipeline_mode=vmem_single_buffer,
    )
    val_spec_fn = lambda e_blk: (e_blk,)

  if weighted:
    val_spec = pl.BlockSpec(
        (edge_block_size,), val_spec_fn, memory_space=smem_space
    )
    in_specs = [row_spec, col_spec, val_spec, x_spec]
    args = [row_ids_aligned, col_ids_aligned, values_aligned, x_3d]
  else:
    in_specs = [row_spec, col_spec, x_spec]
    args = [row_ids_aligned, col_ids_aligned, x_3d]

  scratch_shapes = (
      [pltpu.VMEM((num_rows, m_tile, l_tile), jnp.float32)]
      if use_fp32_acc
      else ()
  )

  return pl.pallas_call(
      kernel_fn,
      out_shape=out_shape,
      grid=grid,
      in_specs=in_specs,
      out_specs=y_spec,
      scratch_shapes=scratch_shapes,
      compiler_params=compiler_params,
      interpret=interpret,
      name='pallas_spmm_kernel',
  )(*args)


def _pick_large_node_tiles(
    n_in: int,
    num_rows: int,
    m_tile: int,
    l_tile: int,
    dtype: Any,
    device_profile: TPUDeviceProfile | str | None = None,
) -> tuple[int, int, int, int]:
  """Selects (b_src, num_src_blocks, b_dst, num_dst_blocks) for spmm_large.

  Maximizes `b_dst` (within `profile.large_avail_vmem_bytes` and `<= 32768`) to
  minimize `num_dst_blocks` (the number of passes streaming `x` from HBM) while
  keeping `b_src <= 32768` and `b_dst <= 32768` so local coordinates
  `(r_local, c_local)` pack losslessly into a single 32-bit `int32` word, and
  adjusts `num_dst_blocks` by +1 when doing so keeps `num_tiles <= 256` (`uint8`
  1-pass radix sort).

  Args:
    n_in: Number of source nodes.
    num_rows: Number of destination nodes.
    m_tile: Sublane tile size.
    l_tile: Lane tile size.
    dtype: Feature dtype.
    device_profile: Optional `TPUDeviceProfile` or profile name.

  Returns:
    Tuple `(b_src, num_src_blocks, b_dst, num_dst_blocks)`.
  """
  profile = get_tpu_device_profile(device_profile)
  d_tile = m_tile * l_tile
  itemsize = jnp.dtype(dtype).itemsize
  use_fp32_acc = itemsize < 4
  src_node_bytes = d_tile * itemsize
  dst_node_bytes = d_tile * (4 if use_fp32_acc else itemsize)

  avail_budget = profile.large_avail_vmem_bytes
  if _MAX_WORKING_SET_BYTES != _DEFAULT_MAX_WORKING_SET_BYTES:
    avail_budget = min(avail_budget, _MAX_WORKING_SET_BYTES)

  stage_bytes = (
      min(_LARGE_STAGE_NODES * d_tile * itemsize, avail_budget // 8)
      if use_fp32_acc
      else 0
  )
  avail_vmem = max(
      src_node_bytes + dst_node_bytes,
      avail_budget - stage_bytes,
  )
  max_coord = 32768

  min_src_bytes = min(
      n_in * src_node_bytes,
      max(src_node_bytes, min(11 * 1024 * 1024, avail_vmem // 4)),
  )
  target_dst_bytes = max(dst_node_bytes, avail_vmem - min_src_bytes)
  max_b_dst = min(max_coord, max(1, target_dst_bytes // dst_node_bytes))
  if num_rows <= max_b_dst:
    b_dst = num_rows
    num_dst_blocks = 1
  else:
    num_dst_blocks = (num_rows + max_b_dst - 1) // max_b_dst
    raw_b_dst = (num_rows + num_dst_blocks - 1) // num_dst_blocks
    align_d = 64 if max_b_dst >= 64 else 1
    b_dst = min(max_coord, ((raw_b_dst + align_d - 1) // align_d) * align_d)
    if b_dst > max_b_dst:
      b_dst = max(1, (max_b_dst // align_d) * align_d)
      num_dst_blocks = (num_rows + b_dst - 1) // b_dst

  rem_src_bytes = max(src_node_bytes, avail_vmem - b_dst * dst_node_bytes)
  max_b_src = min(max_coord, max(1, rem_src_bytes // src_node_bytes))
  if n_in <= max_b_src:
    b_src = n_in
    num_src_blocks = 1
  else:
    num_src_blocks = (n_in + max_b_src - 1) // max_b_src
    raw_b_src = (n_in + num_src_blocks - 1) // num_src_blocks
    align_s = 64 if max_b_src >= 64 else 1
    b_src = min(max_coord, ((raw_b_src + align_s - 1) // align_s) * align_s)
    if b_src > max_b_src:
      b_src = max(1, (max_b_src // align_s) * align_s)
      num_src_blocks = (n_in + b_src - 1) // b_src

  if num_dst_blocks * num_src_blocks > 256 and num_rows > b_dst:
    cand_dst_blocks = num_dst_blocks + 1
    raw_cand_dst = (num_rows + cand_dst_blocks - 1) // cand_dst_blocks
    align_d = 64 if max_b_dst >= 64 else 1
    cand_b_dst = min(
        max_coord, ((raw_cand_dst + align_d - 1) // align_d) * align_d
    )
    cand_rem_src = max(src_node_bytes, avail_vmem - cand_b_dst * dst_node_bytes)
    cand_max_src = min(max_coord, max(1, cand_rem_src // src_node_bytes))
    cand_src_blocks = (n_in + cand_max_src - 1) // cand_max_src
    if cand_dst_blocks * cand_src_blocks <= 256:
      raw_cand_src = (n_in + cand_src_blocks - 1) // cand_src_blocks
      align_s = 64 if cand_max_src >= 64 else 1
      cand_b_src = min(
          max_coord, ((raw_cand_src + align_s - 1) // align_s) * align_s
      )
      if cand_b_src > cand_max_src:
        cand_b_src = max(1, (cand_max_src // align_s) * align_s)
        cand_src_blocks = (n_in + cand_b_src - 1) // cand_b_src
      if cand_dst_blocks * cand_src_blocks <= 256:
        b_dst, num_dst_blocks = cand_b_dst, cand_dst_blocks
        b_src, num_src_blocks = cand_b_src, cand_src_blocks

  return b_src, num_src_blocks, b_dst, num_dst_blocks


@functools.cache
def _make_spmm_large_kernel(
    num_rows: int,
    n_in: int,
    b_dst: int,
    num_dst_blocks: int,
    b_src: int,
    num_src_blocks: int,
    m_tile: int,
    l_tile: int,
    edge_block_size: int,
    unroll: int,
    weighted: bool = False,
    use_fp32_acc: bool = False,
    interpret: bool = False,
):
  """Builds single-pallas_call 2D-bucketed burst-DMA kernel (`spmm_large`)."""
  del num_dst_blocks
  full_dst_blocks = num_rows // b_dst
  rem_dst = num_rows - full_dst_blocks * b_dst
  full_src_blocks = n_in // b_src
  rem_src = n_in - full_src_blocks * b_src
  stage_nodes = min(_LARGE_STAGE_NODES, b_dst)

  def _kernel(starts_smem, ends_smem, packed_rc_hbm, *rest):
    idx = 0
    if weighted:
      val_hbm = rest[idx]
      idx += 1
    else:
      val_hbm = None
    x_hbm = rest[idx]
    y_hbm = rest[idx + 1]
    packed_rc_smem = rest[idx + 2]
    loaded_blk_smem = rest[idx + 3]
    idx += 4
    if weighted:
      val_smem = rest[idx]
      idx += 1
    else:
      val_smem = None
    x_vmem = rest[idx]
    acc_vmem = rest[idx + 1]
    idx += 2
    if use_fp32_acc:
      y_stage_vmem = rest[idx]
      idx += 1
    else:
      y_stage_vmem = None
    dma_sem = rest[idx] if not interpret else None

    need_acc_cast = use_fp32_acc or (acc_vmem.dtype != x_vmem.dtype)

    def _copy_slice(src_ref, src_start, dst_ref, dst_start, length: int):
      if interpret:
        dst_ref[pl.ds(dst_start, length)] = src_ref[pl.ds(src_start, length)]
      else:
        pltpu.async_copy(
            src_ref.at[pl.ds(src_start, length)],
            dst_ref.at[pl.ds(dst_start, length)],
            dma_sem,
        ).wait()

    def _zero_acc(d_len: int):
      z_chunk_len = min(_LARGE_ZERO_CHUNK_NODES, d_len)
      n_zero = d_len // z_chunk_len
      rem_z = d_len % z_chunk_len
      if n_zero > 0:
        z_chunk = jnp.zeros((z_chunk_len, m_tile, l_tile), dtype=acc_vmem.dtype)

        def _z_body(i, _):
          acc_vmem[pl.ds(i * z_chunk_len, z_chunk_len)] = z_chunk
          return None

        jax.lax.fori_loop(0, n_zero, _z_body, None)
      if rem_z > 0:
        acc_vmem[pl.ds(n_zero * z_chunk_len, rem_z)] = jnp.zeros(
            (rem_z, m_tile, l_tile), dtype=acc_vmem.dtype
        )

    def _flush_acc_to_y(d_b, d_len: int):
      y_base = d_b * b_dst
      if not use_fp32_acc:
        _copy_slice(acc_vmem, 0, y_hbm, y_base, d_len)
        return
      assert y_stage_vmem is not None
      s_chunk = min(stage_nodes, d_len)
      n_stage = d_len // s_chunk
      rem_s = d_len % s_chunk
      if n_stage > 0:

        def _stage_body(i, _):
          off = i * s_chunk
          y_stage_vmem[pl.ds(0, s_chunk)] = acc_vmem[
              pl.ds(off, s_chunk)
          ].astype(y_hbm.dtype)
          _copy_slice(y_stage_vmem, 0, y_hbm, y_base + off, s_chunk)
          return None

        jax.lax.fori_loop(0, n_stage, _stage_body, None)
      if rem_s > 0:
        off_rem = n_stage * s_chunk
        y_stage_vmem[pl.ds(0, rem_s)] = acc_vmem[pl.ds(off_rem, rem_s)].astype(
            y_hbm.dtype
        )
        _copy_slice(y_stage_vmem, 0, y_hbm, y_base + off_rem, rem_s)

    def _run_tile(d_b, s_b, s_len: int):
      t_idx = d_b * num_src_blocks + s_b
      start_t = starts_smem[t_idx]
      end_t = ends_smem[t_idx]

      @pl.when(end_t > start_t)
      def _active_tile():
        _copy_slice(x_hbm, s_b * b_src, x_vmem, 0, s_len)
        first_blk = jax.lax.div(start_t, edge_block_size)
        last_blk = jax.lax.div(end_t + edge_block_size - 1, edge_block_size)

        def _blk_body(b, _):
          @pl.when(loaded_blk_smem[0] != b)
          def _load_edge_blk():
            _copy_slice(
                packed_rc_hbm,
                b * edge_block_size,
                packed_rc_smem,
                0,
                edge_block_size,
            )
            if weighted and val_hbm is not None and val_smem is not None:
              _copy_slice(
                  val_hbm,
                  b * edge_block_size,
                  val_smem,
                  0,
                  edge_block_size,
              )
            loaded_blk_smem[0] = b

          blk_base = b * edge_block_size
          lo = jnp.maximum(start_t - blk_base, 0)
          hi = jnp.minimum(end_t - blk_base, edge_block_size)
          count = hi - lo
          n_unroll = jax.lax.div(count, unroll)
          rem = count - n_unroll * unroll

          def _unroll_body(step, _):
            base = lo + step * unroll
            if unroll == 1:
              rc0 = packed_rc_smem[base]
              dst0 = jax.lax.shift_right_logical(rc0, 16)
              src0 = jnp.bitwise_and(rc0, 0xFFFF)
              v0 = x_vmem[pl.ds(src0, 1)]
              if weighted and val_smem is not None:
                v0 = v0 * val_smem[base].astype(x_vmem.dtype)
              if need_acc_cast:
                v0 = v0.astype(acc_vmem.dtype)
              acc_vmem[pl.ds(dst0, 1)] = acc_vmem[pl.ds(dst0, 1)] + v0
              return None

            rcs = [
                packed_rc_smem[base if k == 0 else base + k]
                for k in range(unroll)
            ]
            dsts = [
                jax.lax.shift_right_logical(rcs[k], 16) for k in range(unroll)
            ]
            srcs = [jnp.bitwise_and(rcs[k], 0xFFFF) for k in range(unroll)]

            vs = [x_vmem[pl.ds(srcs[k], 1)] for k in range(unroll)]
            if weighted and val_smem is not None:
              ws = [
                  val_smem[base if k == 0 else base + k].astype(x_vmem.dtype)
                  for k in range(unroll)
              ]
              vs = [vs[k] * ws[k] for k in range(unroll)]
            if need_acc_cast:
              vs = [vs[k].astype(acc_vmem.dtype) for k in range(unroll)]

            us = [acc_vmem[pl.ds(dsts[k], 1)] for k in range(unroll)]

            outs = []
            for k in range(unroll):
              acc = us[k]
              for j in range(k):
                acc = jnp.where(dsts[k] == dsts[j], outs[j], acc)
              outs.append(acc + vs[k])

            for k in range(unroll):
              acc_vmem[pl.ds(dsts[k], 1)] = outs[k]
            return None

          jax.lax.fori_loop(0, n_unroll, _unroll_body, None)

          if unroll > 1:
            tail_base = lo + n_unroll * unroll
            for r in range(unroll - 1):

              @pl.when(r < rem)
              def _do_tail(r=r):
                e_idx = tail_base + r
                rc = packed_rc_smem[e_idx]
                dst_r = jax.lax.shift_right_logical(rc, 16)
                src_r = jnp.bitwise_and(rc, 0xFFFF)
                vr = x_vmem[pl.ds(src_r, 1)]
                if weighted and val_smem is not None:
                  vr = vr * val_smem[e_idx].astype(x_vmem.dtype)
                if need_acc_cast:
                  vr = vr.astype(acc_vmem.dtype)
                acc_vmem[pl.ds(dst_r, 1)] = acc_vmem[pl.ds(dst_r, 1)] + vr

          return None

        jax.lax.fori_loop(first_blk, last_blk, _blk_body, None)

    def _eval_dst_block(d_b, d_len: int):
      _zero_acc(d_len)
      if full_src_blocks > 0:

        def _src_body(s_b, _):
          _run_tile(d_b, s_b, b_src)
          return None

        jax.lax.fori_loop(0, full_src_blocks, _src_body, None)
      if rem_src > 0:
        _run_tile(d_b, full_src_blocks, rem_src)
      _flush_acc_to_y(d_b, d_len)

    loaded_blk_smem[0] = jnp.int32(-1)
    if full_dst_blocks > 0:

      def _dst_body(d_b, _):
        _eval_dst_block(d_b, b_dst)
        return None

      jax.lax.fori_loop(0, full_dst_blocks, _dst_body, None)
    if rem_dst > 0:
      _eval_dst_block(full_dst_blocks, rem_dst)

  return _kernel


def _run_spmm_large_3d(
    x_3d: jax.Array,
    packed_rc_aligned: jax.Array,
    tile_starts_pad: jax.Array,
    tile_ends_pad: jax.Array,
    values_aligned: jax.Array | None,
    num_rows: int,
    b_dst: int,
    num_dst_blocks: int,
    b_src: int,
    num_src_blocks: int,
    m_tile: int,
    l_tile: int,
    edge_block_size: int,
    unroll: int,
    interpret: bool,
    vmem_limit_bytes: int = _DEFAULT_VMEM_LIMIT_BYTES,
) -> jax.Array:
  """Executes single-pallas_call `spmm_large` on a 3D feature tensor."""
  n_in = x_3d.shape[0]
  weighted = values_aligned is not None
  use_fp32_acc = jnp.dtype(x_3d.dtype).itemsize < 4
  num_tiles_pad = tile_starts_pad.shape[0]
  stage_nodes = min(_LARGE_STAGE_NODES, b_dst)

  kernel_fn = _make_spmm_large_kernel(
      num_rows=num_rows,
      n_in=n_in,
      b_dst=b_dst,
      num_dst_blocks=num_dst_blocks,
      b_src=b_src,
      num_src_blocks=num_src_blocks,
      m_tile=m_tile,
      l_tile=l_tile,
      edge_block_size=edge_block_size,
      unroll=unroll,
      weighted=weighted,
      use_fp32_acc=use_fp32_acc,
      interpret=interpret,
  )

  smem_space = None if interpret else pltpu.SMEM
  hbm_space = None if interpret else pl.ANY
  if interpret:
    compiler_params = None
  else:
    compiler_params = pltpu.CompilerParams(
        dimension_semantics=('arbitrary',),
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=True,
    )

  starts_spec = pl.BlockSpec(
      (num_tiles_pad,), lambda _: (0,), memory_space=smem_space
  )
  ends_spec = pl.BlockSpec(
      (num_tiles_pad,), lambda _: (0,), memory_space=smem_space
  )
  hbm_spec = pl.BlockSpec(memory_space=hbm_space)

  if weighted:
    in_specs = [starts_spec, ends_spec, hbm_spec, hbm_spec, hbm_spec]
    args = [
        tile_starts_pad,
        tile_ends_pad,
        packed_rc_aligned,
        values_aligned,
        x_3d,
    ]
  else:
    in_specs = [starts_spec, ends_spec, hbm_spec, hbm_spec]
    args = [tile_starts_pad, tile_ends_pad, packed_rc_aligned, x_3d]

  acc_dtype = jnp.float32 if use_fp32_acc else x_3d.dtype
  scratch_shapes: list[Any] = [
      pltpu.SMEM((edge_block_size,), jnp.int32),
      pltpu.SMEM((1,), jnp.int32),
  ]
  if weighted and values_aligned is not None:
    scratch_shapes.append(pltpu.SMEM((edge_block_size,), values_aligned.dtype))
  scratch_shapes.extend([
      pltpu.VMEM((b_src, m_tile, l_tile), x_3d.dtype),
      pltpu.VMEM((b_dst, m_tile, l_tile), acc_dtype),
  ])
  if use_fp32_acc:
    scratch_shapes.append(
        pltpu.VMEM((stage_nodes, m_tile, l_tile), x_3d.dtype)
    )
  if not interpret:
    scratch_shapes.append(pltpu.SemaphoreType.DMA(()))

  return pl.pallas_call(
      kernel_fn,
      out_shape=jax.ShapeDtypeStruct((num_rows, m_tile, l_tile), x_3d.dtype),
      grid=(1,),
      in_specs=in_specs,
      out_specs=hbm_spec,
      scratch_shapes=scratch_shapes,
      compiler_params=compiler_params,
      interpret=interpret,
      name='pallas_spmm_large_kernel',
  )(*args)


def _run_spmm_large_2d(
    x_2d: jax.Array,
    row_ids: jax.Array,
    col_ids: jax.Array,
    num_rows: int,
    values: jax.Array | None = None,
    *,
    edge_block_size: int = _DEFAULT_LARGE_EDGE_BLOCK_SIZE,
    unroll: int = _DEFAULT_UNROLL,
    interpret: bool = False,
    diag_weight: float = 0.0,
    device_profile: TPUDeviceProfile | str | None = None,
) -> jax.Array:
  """Executes `spmm_large` on a 2D feature matrix [n_in, feat_dim]."""
  n_in, feat_dim = x_2d.shape
  num_edges = row_ids.shape[0]
  if num_edges == 0 or n_in == 0 or num_rows == 0 or feat_dim == 0:
    y_zero = jnp.zeros((num_rows, feat_dim), dtype=x_2d.dtype)
    if diag_weight != 0.0 and num_rows == n_in:
      return y_zero + x_2d * jnp.asarray(diag_weight, dtype=x_2d.dtype)
    return y_zero

  if jax.default_backend() != 'tpu' and not interpret:
    return _cpu_chunked_spmm(
        x_2d, row_ids, col_ids, num_rows, values, diag_weight=diag_weight
    )

  profile = get_tpu_device_profile(device_profile)
  row_ids = row_ids.astype(jnp.int32)
  col_ids = col_ids.astype(jnp.int32)

  if interpret:
    edge_block_size = min(edge_block_size, max(unroll, 64))
  else:
    edge_block_size = max(128, ((edge_block_size + 127) // 128) * 128)
  edge_block_size = max(unroll, (edge_block_size // unroll) * unroll)

  # Prefer l_tile=128 (full sublane packing) for spmm_large.
  itemsize = jnp.dtype(x_2d.dtype).itemsize
  packing = max(1, 4 // itemsize)
  valid_m = [m for m in (8, 4, 2, 1) if m >= packing]
  m_tile, l_tile, d_pad = 8, 128, ((feat_dim + 1023) // 1024) * 1024
  for m in valid_m:
    if feat_dim % (m * 128) == 0 and (feat_dim // m) <= 512:
      m_tile, l_tile, d_pad = m, feat_dim // m, feat_dim
      break
  else:
    for m in reversed(valid_m):
      if feat_dim <= m * 128:
        m_tile, l_tile, d_pad = m, 128, m * 128
        break

  if d_pad > feat_dim:
    x_padded = jnp.pad(x_2d, ((0, 0), (0, d_pad - feat_dim)))
  else:
    x_padded = x_2d

  max_chunk_dim = profile.max_large_chunk_dim
  if d_pad <= max_chunk_dim:
    chunk_m_tile, chunk_l_tile, chunk_dim = m_tile, l_tile, d_pad
    g_col = 1
  else:
    chunk_m_tile, chunk_l_tile, chunk_dim = pick_3d_tile_shape(
        max_chunk_dim, x_2d.dtype
    )
    if chunk_dim == 256 and chunk_m_tile == 1:
      chunk_m_tile, chunk_l_tile = 2, 128
    if d_pad % chunk_dim != 0:
      d_pad = ((d_pad + chunk_dim - 1) // chunk_dim) * chunk_dim
      x_padded = jnp.pad(x_2d, ((0, 0), (0, d_pad - feat_dim)))
    g_col = d_pad // chunk_dim

  b_src, num_src_blocks, b_dst, num_dst_blocks = _pick_large_node_tiles(
      n_in,
      num_rows,
      chunk_m_tile,
      chunk_l_tile,
      x_2d.dtype,
      device_profile=profile,
  )
  num_tiles = num_dst_blocks * num_src_blocks

  # Bucket edges by 2D (dst_blk, src_blk) tile and pack 16-bit local coords.
  r_clamped = jnp.clip(row_ids, 0, num_rows - 1)
  c_clamped = jnp.clip(col_ids, 0, n_in - 1)
  d_b = jax.lax.div(r_clamped, b_dst)
  s_b = jax.lax.div(c_clamped, b_src)
  r_local = r_clamped - d_b * b_dst
  c_local = c_clamped - s_b * b_src
  tile_id = d_b * num_src_blocks + s_b
  packed_rc = jnp.bitwise_or(
      jnp.left_shift(r_local, 16), jnp.bitwise_and(c_local, 0xFFFF)
  )

  # Use narrower key dtype (uint8 when num_tiles <= 256, uint16 when <= 65536)
  # and unstable radix sort on large graphs to reduce radix sort passes by 2-4x.
  if num_tiles <= 256:
    key_dtype = jnp.uint8
  elif num_tiles <= 65536:
    key_dtype = jnp.uint16
  else:
    key_dtype = jnp.int32
  tile_key = tile_id.astype(key_dtype)
  use_stable_sort = interpret or num_edges <= 16384

  if values is None:
    tile_sorted, packed_rc_sorted = jax.lax.sort(
        [tile_key, packed_rc],
        dimension=0,
        is_stable=use_stable_sort,
        num_keys=1,
    )
    values_sorted = None
  else:
    tile_sorted, packed_rc_sorted, values_sorted = jax.lax.sort(
        [tile_key, packed_rc, values],
        dimension=0,
        is_stable=use_stable_sort,
        num_keys=1,
    )

  tile_ids_arange = jnp.arange(num_tiles, dtype=key_dtype)
  tile_starts = jnp.searchsorted(
      tile_sorted, tile_ids_arange, side='left'
  ).astype(jnp.int32)
  tile_ends = jnp.concatenate([
      tile_starts[1:],
      jnp.array([num_edges], dtype=jnp.int32),
  ])

  num_tiles_pad = max(8, ((num_tiles + 7) // 8) * 8)
  pad_tiles = num_tiles_pad - num_tiles
  if pad_tiles > 0:
    tile_starts_pad = jnp.pad(
        tile_starts, (0, pad_tiles), constant_values=num_edges
    )
    tile_ends_pad = jnp.pad(
        tile_ends, (0, pad_tiles), constant_values=num_edges
    )
  else:
    tile_starts_pad = tile_starts
    tile_ends_pad = tile_ends

  num_edge_blocks = (num_edges + edge_block_size - 1) // edge_block_size
  e_aligned = num_edge_blocks * edge_block_size
  pad_edges = e_aligned - num_edges
  if pad_edges > 0:
    packed_rc_aligned = jnp.pad(
        packed_rc_sorted, (0, pad_edges), constant_values=0
    )
    values_aligned = (
        jnp.pad(values_sorted, (0, pad_edges), constant_values=0)
        if values_sorted is not None
        else None
    )
  else:
    packed_rc_aligned = packed_rc_sorted
    values_aligned = values_sorted

  if g_col == 1:
    x_3d = x_padded.reshape(n_in, chunk_m_tile, chunk_l_tile)
    y_3d = _run_spmm_large_3d(
        x_3d=x_3d,
        packed_rc_aligned=packed_rc_aligned,
        tile_starts_pad=tile_starts_pad,
        tile_ends_pad=tile_ends_pad,
        values_aligned=values_aligned,
        num_rows=num_rows,
        b_dst=b_dst,
        num_dst_blocks=num_dst_blocks,
        b_src=b_src,
        num_src_blocks=num_src_blocks,
        m_tile=chunk_m_tile,
        l_tile=chunk_l_tile,
        edge_block_size=edge_block_size,
        unroll=unroll,
        interpret=interpret,
        vmem_limit_bytes=profile.vmem_limit_bytes,
    )
    y_2d = y_3d.reshape(num_rows, d_pad)
  else:
    x_chunks = jnp.swapaxes(
        x_padded.reshape(n_in, g_col, chunk_m_tile, chunk_l_tile), 0, 1
    )

    def _eval_col_chunk(x_chunk: jax.Array) -> jax.Array:
      return _run_spmm_large_3d(
          x_3d=x_chunk,
          packed_rc_aligned=packed_rc_aligned,
          tile_starts_pad=tile_starts_pad,
          tile_ends_pad=tile_ends_pad,
          values_aligned=values_aligned,
          num_rows=num_rows,
          b_dst=b_dst,
          num_dst_blocks=num_dst_blocks,
          b_src=b_src,
          num_src_blocks=num_src_blocks,
          m_tile=chunk_m_tile,
          l_tile=chunk_l_tile,
          edge_block_size=edge_block_size,
          unroll=unroll,
          interpret=interpret,
          vmem_limit_bytes=profile.vmem_limit_bytes,
      )

    y_chunks = jax.lax.map(_eval_col_chunk, x_chunks)
    y_2d = jnp.swapaxes(y_chunks, 0, 1).reshape(num_rows, d_pad)

  if d_pad > feat_dim:
    y_2d = y_2d[:, :feat_dim]
  if diag_weight != 0.0 and num_rows == n_in:
    y_2d = y_2d + x_2d * jnp.asarray(diag_weight, dtype=x_2d.dtype)
  return y_2d


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7))
def _spmm_large_unweighted(
    x_2d: jax.Array,
    row_ids: jax.Array,
    col_ids: jax.Array,
    num_rows: int,
    edge_block_size: int,
    unroll: int,
    interpret: bool,
    device_profile: TPUDeviceProfile | str | None = None,
) -> jax.Array:
  """Unweighted 2D `spmm_large` with custom VJP."""
  return _run_spmm_large_2d(
      x_2d,
      row_ids,
      col_ids,
      num_rows,
      None,
      edge_block_size=edge_block_size,
      unroll=unroll,
      interpret=interpret,
      device_profile=device_profile,
  )


def _spmm_large_unweighted_fwd(
    x_2d: jax.Array,
    row_ids: jax.Array,
    col_ids: jax.Array,
    num_rows: int,
    edge_block_size: int,
    unroll: int,
    interpret: bool,
    device_profile: TPUDeviceProfile | str | None = None,
):
  """Forward pass for `_spmm_large_unweighted`."""
  y_2d = _run_spmm_large_2d(
      x_2d,
      row_ids,
      col_ids,
      num_rows,
      None,
      edge_block_size=edge_block_size,
      unroll=unroll,
      interpret=interpret,
      device_profile=device_profile,
  )
  return y_2d, (row_ids, col_ids, x_2d.shape[0])


def _spmm_large_unweighted_bwd(
    num_rows: int,
    edge_block_size: int,
    unroll: int,
    interpret: bool,
    device_profile: TPUDeviceProfile | str | None,
    residuals,
    dy_2d: jax.Array,
):
  """Backward pass for `_spmm_large_unweighted`."""
  del num_rows
  row_ids, col_ids, n_in = residuals
  dx_2d = _run_spmm_large_2d(
      dy_2d,
      col_ids,
      row_ids,
      n_in,
      None,
      edge_block_size=edge_block_size,
      unroll=unroll,
      interpret=interpret,
      device_profile=device_profile,
  )
  return dx_2d, None, None


_spmm_large_unweighted.defvjp(
    _spmm_large_unweighted_fwd, _spmm_large_unweighted_bwd
)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8))
def _spmm_large_weighted(
    x_2d: jax.Array,
    row_ids: jax.Array,
    col_ids: jax.Array,
    values: jax.Array,
    num_rows: int,
    edge_block_size: int,
    unroll: int,
    interpret: bool,
    device_profile: TPUDeviceProfile | str | None = None,
) -> jax.Array:
  """Weighted 2D `spmm_large` with custom VJP wrt x_2d."""
  return _run_spmm_large_2d(
      x_2d,
      row_ids,
      col_ids,
      num_rows,
      values,
      edge_block_size=edge_block_size,
      unroll=unroll,
      interpret=interpret,
      device_profile=device_profile,
  )


def _spmm_large_weighted_fwd(
    x_2d: jax.Array,
    row_ids: jax.Array,
    col_ids: jax.Array,
    values: jax.Array,
    num_rows: int,
    edge_block_size: int,
    unroll: int,
    interpret: bool,
    device_profile: TPUDeviceProfile | str | None = None,
):
  """Forward pass for `_spmm_large_weighted`."""
  y_2d = _run_spmm_large_2d(
      x_2d,
      row_ids,
      col_ids,
      num_rows,
      values,
      edge_block_size=edge_block_size,
      unroll=unroll,
      interpret=interpret,
      device_profile=device_profile,
  )
  return y_2d, (row_ids, col_ids, values, x_2d.shape[0])


def _spmm_large_weighted_bwd(
    num_rows: int,
    edge_block_size: int,
    unroll: int,
    interpret: bool,
    device_profile: TPUDeviceProfile | str | None,
    residuals,
    dy_2d: jax.Array,
):
  """Backward pass for `_spmm_large_weighted`."""
  del num_rows
  row_ids, col_ids, values, n_in = residuals
  dx_2d = _run_spmm_large_2d(
      dy_2d,
      col_ids,
      row_ids,
      n_in,
      values,
      edge_block_size=edge_block_size,
      unroll=unroll,
      interpret=interpret,
      device_profile=device_profile,
  )
  return dx_2d, None, None, None


_spmm_large_weighted.defvjp(_spmm_large_weighted_fwd, _spmm_large_weighted_bwd)


def spmm_large(
    x: jax.Array,
    row_ids: jax.Array,
    col_ids: jax.Array,
    num_rows: int,
    values: jax.Array | None = None,
    *,
    edge_block_size: int = _DEFAULT_LARGE_EDGE_BLOCK_SIZE,
    unroll: int = _DEFAULT_UNROLL,
    interpret: bool = False,
    device_profile: TPUDeviceProfile | str | None = None,
) -> jax.Array:
  """Computes O(N)-memory SpMM for large-N graphs exceeding VMEM capacity."""
  orig_shape = x.shape
  n_in = orig_shape[0]
  trailing_shape = orig_shape[1:]
  feat_dim = math.prod(trailing_shape) if trailing_shape else 1
  x_2d = x.reshape(n_in, feat_dim)

  if values is None:
    y_2d = _spmm_large_unweighted(
        x_2d,
        row_ids,
        col_ids,
        num_rows,
        edge_block_size,
        unroll,
        interpret,
        device_profile,
    )
  else:
    y_2d = _spmm_large_weighted(
        x_2d,
        row_ids,
        col_ids,
        values,
        num_rows,
        edge_block_size,
        unroll,
        interpret,
        device_profile,
    )
  return y_2d.reshape((num_rows, *trailing_shape))


def _run_pallas_spmm_2d(
    x_2d: jax.Array,
    row_ids: jax.Array,
    col_ids: jax.Array,
    num_rows: int,
    values: jax.Array | None = None,
    *,
    edge_block_size: int = _DEFAULT_EDGE_BLOCK_SIZE,
    unroll: int = _DEFAULT_UNROLL,
    interpret: bool = False,
    diag_weight: float = 0.0,
    device_profile: TPUDeviceProfile | str | None = None,
) -> jax.Array:
  """Executes direct-edge Pallas SpMM on a 2D feature matrix x_2d."""
  n_in, feat_dim = x_2d.shape
  num_edges = row_ids.shape[0]
  if num_edges == 0 or n_in == 0 or num_rows == 0 or feat_dim == 0:
    y_zero = jnp.zeros((num_rows, feat_dim), dtype=x_2d.dtype)
    if diag_weight != 0.0 and num_rows == n_in:
      return y_zero + x_2d * jnp.asarray(diag_weight, dtype=x_2d.dtype)
    return y_zero

  if jax.default_backend() != 'tpu' and not interpret:
    return _cpu_chunked_spmm(
        x_2d, row_ids, col_ids, num_rows, values, diag_weight=diag_weight
    )

  profile = get_tpu_device_profile(device_profile)
  max_working_set = _effective_max_working_set_bytes(profile)

  m_tile, l_tile, d_pad = pick_3d_tile_shape(feat_dim, x_2d.dtype)
  itemsize = jnp.dtype(x_2d.dtype).itemsize
  acc_bytes_per_elem = 4 if itemsize < 4 else 0
  bytes_per_col = n_in * itemsize + num_rows * (itemsize + acc_bytes_per_elem)
  f32_bytes_per_col = (n_in + num_rows) * 4
  working_set_bytes = bytes_per_col * d_pad

  min_1d_tile_bytes = min(
      bytes_per_col * min(d_pad, 256),
      f32_bytes_per_col * min(d_pad, 128),
  )
  if min_1d_tile_bytes > max_working_set:
    large_blk = (
        _DEFAULT_LARGE_EDGE_BLOCK_SIZE
        if edge_block_size == _DEFAULT_EDGE_BLOCK_SIZE and not interpret
        else edge_block_size
    )
    return _run_spmm_large_2d(
        x_2d,
        row_ids,
        col_ids,
        num_rows,
        values,
        edge_block_size=large_blk,
        unroll=unroll,
        interpret=interpret,
        diag_weight=diag_weight,
        device_profile=profile,
    )

  row_ids = row_ids.astype(jnp.int32)
  col_ids = col_ids.astype(jnp.int32)

  if interpret:
    edge_block_size = min(edge_block_size, max(unroll, 64))
  edge_block_size = max(unroll, (edge_block_size // unroll) * unroll)

  num_edge_blocks = (num_edges + edge_block_size - 1) // edge_block_size
  e_aligned = num_edge_blocks * edge_block_size
  last_block_edges = num_edges - (num_edge_blocks - 1) * edge_block_size
  pad_edges = e_aligned - num_edges

  if pad_edges > 0:
    row_ids_aligned = jnp.pad(row_ids, (0, pad_edges), constant_values=0)
    col_ids_aligned = jnp.pad(col_ids, (0, pad_edges), constant_values=0)
    if values is not None:
      values_aligned = jnp.pad(values, (0, pad_edges), constant_values=0)
    else:
      values_aligned = None
  else:
    row_ids_aligned = row_ids
    col_ids_aligned = col_ids
    values_aligned = values

  if d_pad > feat_dim:
    x_padded = jnp.pad(x_2d, ((0, 0), (0, d_pad - feat_dim)))
  else:
    x_padded = x_2d

  if working_set_bytes <= max_working_set:
    x_3d = x_padded.reshape(n_in, m_tile, l_tile)
    y_3d = _run_single_3d_tile(
        x_3d=x_3d,
        row_ids_aligned=row_ids_aligned,
        col_ids_aligned=col_ids_aligned,
        values_aligned=values_aligned,
        num_rows=num_rows,
        m_tile=m_tile,
        l_tile=l_tile,
        edge_block_size=edge_block_size,
        num_edge_blocks=num_edge_blocks,
        last_block_edges=last_block_edges,
        unroll=unroll,
        interpret=interpret,
        diag_weight=diag_weight,
        vmem_limit_bytes=profile.vmem_limit_bytes,
    )
    y_2d = y_3d.reshape(num_rows, d_pad)
  else:
    packing = max(1, 4 // itemsize)
    upcast_chunk_to_f32 = (
        itemsize < 4 and bytes_per_col * (packing * 128) > max_working_set
    )
    if upcast_chunk_to_f32:
      chunk_m_tile = 1
      chunk_l_tile = 128
      chunk_dim = 128
    else:
      max_chunk_dim = max(packing * 128, max_working_set // bytes_per_col)
      chunk_m_tile = packing
      for cand_m in (8, 4, 2, 1):
        if (
            cand_m >= packing
            and cand_m * 128 <= max_chunk_dim
            and d_pad % (cand_m * 128) == 0
        ):
          chunk_m_tile = cand_m
          break
      chunk_l_tile = 128
      chunk_dim = chunk_m_tile * chunk_l_tile

    if d_pad % chunk_dim != 0:
      d_pad = ((d_pad + chunk_dim - 1) // chunk_dim) * chunk_dim
      x_padded = jnp.pad(x_2d, ((0, 0), (0, d_pad - feat_dim)))
    g_col = d_pad // chunk_dim

    if not upcast_chunk_to_f32 and not interpret:
      x_3d_multi = x_padded.reshape(n_in, chunk_m_tile, g_col * chunk_l_tile)
      y_3d = _run_single_3d_tile(
          x_3d=x_3d_multi,
          row_ids_aligned=row_ids_aligned,
          col_ids_aligned=col_ids_aligned,
          values_aligned=values_aligned,
          num_rows=num_rows,
          m_tile=chunk_m_tile,
          l_tile=chunk_l_tile,
          edge_block_size=edge_block_size,
          num_edge_blocks=num_edge_blocks,
          last_block_edges=last_block_edges,
          unroll=unroll,
          interpret=interpret,
          diag_weight=diag_weight,
          vmem_limit_bytes=profile.vmem_limit_bytes,
          g_col=g_col,
      )
      y_2d = y_3d.reshape(num_rows, d_pad)
    else:
      y_chunks = []
      for c in range(g_col):
        x_slice = x_padded[:, c * chunk_dim : (c + 1) * chunk_dim]
        if upcast_chunk_to_f32:
          x_slice = x_slice.astype(jnp.float32)
        x_chunk = x_slice.reshape(n_in, chunk_m_tile, chunk_l_tile)
        y_chunk = _run_single_3d_tile(
            x_3d=x_chunk,
            row_ids_aligned=row_ids_aligned,
            col_ids_aligned=col_ids_aligned,
            values_aligned=values_aligned,
            num_rows=num_rows,
            m_tile=chunk_m_tile,
            l_tile=chunk_l_tile,
            edge_block_size=edge_block_size,
            num_edge_blocks=num_edge_blocks,
            last_block_edges=last_block_edges,
            unroll=unroll,
            interpret=interpret,
            diag_weight=diag_weight,
            vmem_limit_bytes=profile.vmem_limit_bytes,
        )
        if upcast_chunk_to_f32:
          y_chunk = y_chunk.astype(x_2d.dtype)
        y_chunks.append(y_chunk.reshape(num_rows, chunk_dim))
      y_2d = jnp.concatenate(y_chunks, axis=1)

  if d_pad > feat_dim:
    y_2d = y_2d[:, :feat_dim]
  return y_2d


def _run_lazy_cosort_2d(
    x_2d: jax.Array,
    row_ids: jax.Array,
    col_ids: jax.Array,
    num_rows: int,
    block_size: int,
    unroll: int,
    interpret: bool,
    diag_weight: float = 0.0,
    device_profile: TPUDeviceProfile | str | None = None,
) -> jax.Array:
  """Lazily obtains the single-direction CoSort schedule and runs the kernel."""
  n_in = x_2d.shape[0]
  sched = _get_or_build_single_schedule(
      row_ids, col_ids, int(num_rows), int(n_in), block_size=block_size
  )
  return _run_cosort_spmm_2d(
      x_2d,
      sched,
      int(num_rows),
      block_size=block_size,
      unroll=unroll,
      interpret=interpret,
      diag_weight=diag_weight,
      device_profile=device_profile,
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8))
def _pallas_spmm_cosort_lazy(
    x_2d: jax.Array,
    row_ids: jax.Array,
    col_ids: jax.Array,
    num_rows: int,
    block_size: int,
    unroll: int,
    interpret: bool,
    diag_weight: float = 0.0,
    device_profile: TPUDeviceProfile | str | None = None,
) -> jax.Array:
  """Unweighted 2D Pallas SpMM with lazy per-direction schedule caching."""
  return _run_lazy_cosort_2d(
      x_2d,
      row_ids,
      col_ids,
      num_rows,
      block_size=block_size,
      unroll=unroll,
      interpret=interpret,
      diag_weight=diag_weight,
      device_profile=device_profile,
  )


def _pallas_spmm_cosort_lazy_fwd(
    x_2d: jax.Array,
    row_ids: jax.Array,
    col_ids: jax.Array,
    num_rows: int,
    block_size: int,
    unroll: int,
    interpret: bool,
    diag_weight: float = 0.0,
    device_profile: TPUDeviceProfile | str | None = None,
):
  """Forward pass for `_pallas_spmm_cosort_lazy`."""
  y_2d = _run_lazy_cosort_2d(
      x_2d,
      row_ids,
      col_ids,
      num_rows,
      block_size=block_size,
      unroll=unroll,
      interpret=interpret,
      diag_weight=diag_weight,
      device_profile=device_profile,
  )
  return y_2d, (row_ids, col_ids, x_2d.shape[0])


def _pallas_spmm_cosort_lazy_bwd(
    num_rows: int,
    block_size: int,
    unroll: int,
    interpret: bool,
    diag_weight: float,
    device_profile: TPUDeviceProfile | str | None,
    residuals,
    dy_2d: jax.Array,
):
  """Backward pass for `_pallas_spmm_cosort_lazy`."""
  del num_rows
  row_ids, col_ids, n_in = residuals
  dx_2d = _run_lazy_cosort_2d(
      dy_2d,
      col_ids,
      row_ids,
      n_in,
      block_size=block_size,
      unroll=unroll,
      interpret=interpret,
      diag_weight=diag_weight,
      device_profile=device_profile,
  )
  return dx_2d, None, None


_pallas_spmm_cosort_lazy.defvjp(
    _pallas_spmm_cosort_lazy_fwd, _pallas_spmm_cosort_lazy_bwd
)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8, 9))
def _pallas_spmm_scheduled(
    x_2d: jax.Array,
    sched_fwd: tuple[jax.Array, jax.Array, jax.Array],
    sched_bwd: tuple[jax.Array, jax.Array, jax.Array],
    num_rows: int,
    n_in: int,
    block_size: int,
    unroll: int,
    interpret: bool,
    diag_weight: float = 0.0,
    device_profile: TPUDeviceProfile | str | None = None,
) -> jax.Array:
  """Scheduled 2D Pallas SpMM with precomputed `(sched_fwd, sched_bwd)`."""
  del sched_bwd, n_in
  return _run_cosort_spmm_2d(
      x_2d,
      sched_fwd,
      num_rows,
      block_size=block_size,
      unroll=unroll,
      interpret=interpret,
      diag_weight=diag_weight,
      device_profile=device_profile,
  )


def _pallas_spmm_scheduled_fwd(
    x_2d: jax.Array,
    sched_fwd: tuple[jax.Array, jax.Array, jax.Array],
    sched_bwd: tuple[jax.Array, jax.Array, jax.Array],
    num_rows: int,
    n_in: int,
    block_size: int,
    unroll: int,
    interpret: bool,
    diag_weight: float = 0.0,
    device_profile: TPUDeviceProfile | str | None = None,
):
  """Forward pass for `_pallas_spmm_scheduled`."""
  del n_in
  y_2d = _run_cosort_spmm_2d(
      x_2d,
      sched_fwd,
      num_rows,
      block_size=block_size,
      unroll=unroll,
      interpret=interpret,
      diag_weight=diag_weight,
      device_profile=device_profile,
  )
  return y_2d, sched_bwd


def _pallas_spmm_scheduled_bwd(
    num_rows: int,
    n_in: int,
    block_size: int,
    unroll: int,
    interpret: bool,
    diag_weight: float,
    device_profile: TPUDeviceProfile | str | None,
    sched_bwd: tuple[jax.Array, jax.Array, jax.Array],
    dy_2d: jax.Array,
):
  """Backward pass for `_pallas_spmm_scheduled`."""
  del num_rows
  dx_2d = _run_cosort_spmm_2d(
      dy_2d,
      sched_bwd,
      n_in,
      block_size=block_size,
      unroll=unroll,
      interpret=interpret,
      diag_weight=diag_weight,
      device_profile=device_profile,
  )
  return dx_2d, None, None


_pallas_spmm_scheduled.defvjp(
    _pallas_spmm_scheduled_fwd, _pallas_spmm_scheduled_bwd
)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8))
def _pallas_spmm_unweighted(
    x_2d: jax.Array,
    row_ids: jax.Array,
    col_ids: jax.Array,
    num_rows: int,
    edge_block_size: int,
    unroll: int,
    interpret: bool,
    diag_weight: float = 0.0,
    device_profile: TPUDeviceProfile | str | None = None,
) -> jax.Array:
  """Unweighted 2D Pallas SpMM with custom VJP (direct-edge / spmm_large)."""
  return _run_pallas_spmm_2d(
      x_2d,
      row_ids,
      col_ids,
      num_rows,
      None,
      edge_block_size=edge_block_size,
      unroll=unroll,
      interpret=interpret,
      diag_weight=diag_weight,
      device_profile=device_profile,
  )


def _pallas_spmm_unweighted_fwd(
    x_2d: jax.Array,
    row_ids: jax.Array,
    col_ids: jax.Array,
    num_rows: int,
    edge_block_size: int,
    unroll: int,
    interpret: bool,
    diag_weight: float = 0.0,
    device_profile: TPUDeviceProfile | str | None = None,
):
  """Forward pass for `_pallas_spmm_unweighted`."""
  y_2d = _run_pallas_spmm_2d(
      x_2d,
      row_ids,
      col_ids,
      num_rows,
      None,
      edge_block_size=edge_block_size,
      unroll=unroll,
      interpret=interpret,
      diag_weight=diag_weight,
      device_profile=device_profile,
  )
  return y_2d, (row_ids, col_ids, x_2d.shape[0])


def _pallas_spmm_unweighted_bwd(
    num_rows: int,
    edge_block_size: int,
    unroll: int,
    interpret: bool,
    diag_weight: float,
    device_profile: TPUDeviceProfile | str | None,
    residuals,
    dy_2d: jax.Array,
):
  """Backward pass for `_pallas_spmm_unweighted`."""
  del num_rows
  row_ids, col_ids, n_in = residuals
  dx_2d = _run_pallas_spmm_2d(
      dy_2d,
      col_ids,
      row_ids,
      n_in,
      None,
      edge_block_size=edge_block_size,
      unroll=unroll,
      interpret=interpret,
      diag_weight=diag_weight,
      device_profile=device_profile,
  )
  return dx_2d, None, None


_pallas_spmm_unweighted.defvjp(
    _pallas_spmm_unweighted_fwd, _pallas_spmm_unweighted_bwd
)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8, 9))
def _pallas_spmm_weighted(
    x_2d: jax.Array,
    row_ids: jax.Array,
    col_ids: jax.Array,
    values: jax.Array,
    num_rows: int,
    edge_block_size: int,
    unroll: int,
    interpret: bool,
    diag_weight: float = 0.0,
    device_profile: TPUDeviceProfile | str | None = None,
) -> jax.Array:
  """Weighted 2D Pallas SpMM with custom VJP wrt x_2d."""
  return _run_pallas_spmm_2d(
      x_2d,
      row_ids,
      col_ids,
      num_rows,
      values,
      edge_block_size=edge_block_size,
      unroll=unroll,
      interpret=interpret,
      diag_weight=diag_weight,
      device_profile=device_profile,
  )


def _pallas_spmm_weighted_fwd(
    x_2d: jax.Array,
    row_ids: jax.Array,
    col_ids: jax.Array,
    values: jax.Array,
    num_rows: int,
    edge_block_size: int,
    unroll: int,
    interpret: bool,
    diag_weight: float = 0.0,
    device_profile: TPUDeviceProfile | str | None = None,
):
  """Forward pass for `_pallas_spmm_weighted`."""
  y_2d = _run_pallas_spmm_2d(
      x_2d,
      row_ids,
      col_ids,
      num_rows,
      values,
      edge_block_size=edge_block_size,
      unroll=unroll,
      interpret=interpret,
      diag_weight=diag_weight,
      device_profile=device_profile,
  )
  return y_2d, (row_ids, col_ids, values, x_2d.shape[0])


def _pallas_spmm_weighted_bwd(
    num_rows: int,
    edge_block_size: int,
    unroll: int,
    interpret: bool,
    diag_weight: float,
    device_profile: TPUDeviceProfile | str | None,
    residuals,
    dy_2d: jax.Array,
):
  """Backward pass for `_pallas_spmm_weighted`."""
  del num_rows
  row_ids, col_ids, values, n_in = residuals
  dx_2d = _run_pallas_spmm_2d(
      dy_2d,
      col_ids,
      row_ids,
      n_in,
      values,
      edge_block_size=edge_block_size,
      unroll=unroll,
      interpret=interpret,
      diag_weight=diag_weight,
      device_profile=device_profile,
  )
  return dx_2d, None, None, None


_pallas_spmm_weighted.defvjp(
    _pallas_spmm_weighted_fwd, _pallas_spmm_weighted_bwd
)


@functools.lru_cache(maxsize=64)
def _make_tiled_y_spmm_kernel(
    num_rows: int,
    m_tile: int,
    l_tile: int,
    b_dst: int,
    edge_blk: int,
):
  """Builds a single-pass (g_col=1) Tiled-Y SpMM kernel with full X in VMEM."""
  num_dst_blocks = (num_rows + b_dst - 1) // b_dst
  full_dst_blocks = num_rows // b_dst
  rem_dst = num_rows - full_dst_blocks * b_dst
  z_chunk_len = min(512, b_dst)

  def _tiled_y_spmm_kernel(
      offsets_smem,
      packed_rc_hbm,
      x_vmem,
      y_hbm,
      packed_rc_smem,
      loaded_blk_smem,
      y_tile_vmem,
      dma_sem,
  ):
    z = jnp.zeros((1, m_tile, l_tile), dtype=jnp.float32)
    z_chunk = jnp.zeros((z_chunk_len, m_tile, l_tile), dtype=jnp.float32)
    loaded_blk_smem[0] = jnp.int32(-1)

    def _eval_tile(d_b, d_len):
      for i in range(b_dst // z_chunk_len):
        y_tile_vmem[pl.ds(i * z_chunk_len, z_chunk_len)] = z_chunk
      start_t = offsets_smem[d_b]
      end_t = offsets_smem[d_b + 1]
      d_base = d_b * b_dst

      @pl.when(end_t > start_t)
      def _active():
        first_blk = jax.lax.div(start_t, edge_blk)
        last_blk = jax.lax.div(end_t + edge_blk - 1, edge_blk)

        def _blk_body(b, state):
          prev_d, o = state

          @pl.when(loaded_blk_smem[0] != b)
          def _load_blk():
            pltpu.async_copy(
                packed_rc_hbm.at[pl.ds(b * edge_blk, edge_blk)],
                packed_rc_smem.at[pl.ds(0, edge_blk)],
                dma_sem,
            ).wait()
            loaded_blk_smem[0] = b

          blk_base = b * edge_blk
          lo = jnp.maximum(start_t - blk_base, 0)
          hi = jnp.minimum(end_t - blk_base, edge_blk)
          cnt = hi - lo
          n_steps = jax.lax.div(cnt, 8)
          rem_start = lo + n_steps * 8

          def _step8(step, st):
            p_d, acc = st
            pos = lo + step * 8
            rcs = [
                packed_rc_smem[pos if u == 0 else pos + u] for u in range(8)
            ]
            ds = [jax.lax.shift_right_logical(rcs[u], 16) for u in range(8)]
            ss = [jnp.bitwise_and(rcs[u], 0xFFFF) for u in range(8)]
            vs = [
                x_vmem[pl.ds(ss[u], 1)].astype(jnp.float32) for u in range(8)
            ]
            for u in range(8):
              same = ds[u] == p_d
              y_tile_vmem[pl.ds(p_d, 1)] = acc
              acc = jnp.where(same, acc, z) + vs[u]
              p_d = ds[u]
            return (p_d, acc)

          prev_d1, o1 = jax.lax.fori_loop(0, n_steps, _step8, (prev_d, o))

          def _rem1(idx, st):
            p_d, acc = st
            rc = packed_rc_smem[idx]
            d_u = jax.lax.shift_right_logical(rc, 16)
            s_u = jnp.bitwise_and(rc, 0xFFFF)
            v_u = x_vmem[pl.ds(s_u, 1)].astype(jnp.float32)
            same = d_u == p_d
            y_tile_vmem[pl.ds(p_d, 1)] = acc
            acc = jnp.where(same, acc, z) + v_u
            return (d_u, acc)

          return jax.lax.fori_loop(rem_start, hi, _rem1, (prev_d1, o1))

        final_d, final_o = jax.lax.fori_loop(
            first_blk, last_blk, _blk_body, (jnp.int32(b_dst), z)
        )
        y_tile_vmem[pl.ds(final_d, 1)] = final_o

      @pl.when(d_b == (num_dst_blocks - 1))
      def _apply_last_pad():
        pad_cnt = offsets_smem[num_dst_blocks + 1]

        @pl.when(pad_cnt > 0)
        def _do_pad():
          last_loc = (num_rows - 1) - d_base
          vp = x_vmem[pl.ds(num_rows - 1, 1)].astype(jnp.float32)
          y_tile_vmem[pl.ds(last_loc, 1)] = (
              y_tile_vmem[pl.ds(last_loc, 1)] + vp * pad_cnt.astype(jnp.float32)
          )

      pltpu.async_copy(
          y_tile_vmem.at[pl.ds(0, d_len)],
          y_hbm.at[pl.ds(d_base, d_len)],
          dma_sem,
      ).wait()

    if full_dst_blocks > 0:
      def _dst_loop(d_b, _):
        _eval_tile(d_b, b_dst)
        return None

      jax.lax.fori_loop(0, full_dst_blocks, _dst_loop, None)
    if rem_dst > 0:
      _eval_tile(full_dst_blocks, rem_dst)

  return _tiled_y_spmm_kernel


def _run_spmm_tiled_y_2d(
    x_2d: jax.Array,
    row_ids: jax.Array,
    col_ids: jax.Array,
    num_rows: int,
    profile: TPUDeviceProfile,
) -> jax.Array:
  """Executes single-sort g_col=1 Tiled-Y SpMM on TPU."""
  n_in, feat_dim = x_2d.shape
  orig_dtype = x_2d.dtype
  m_tile, l_tile, d_pad = pick_3d_tile_shape(feat_dim, orig_dtype)
  if feat_dim < d_pad:
    x_pad = jnp.pad(x_2d, ((0, 0), (0, d_pad - feat_dim)))
  else:
    x_pad = x_2d

  itemsize = jnp.dtype(orig_dtype).itemsize
  x_vmem_bytes = n_in * d_pad * itemsize
  if x_vmem_bytes + (4096 + 4) * d_pad * 4 <= (
      profile.vmem_limit_bytes - 1024 * 1024
  ):
    b_dst = 4096
  else:
    b_dst = 2048
  edge_blk = 16384

  e_len = row_ids.shape[0]
  n_blks = max(1, (e_len + edge_blk - 1) // edge_blk)
  e_aligned = n_blks * edge_blk
  e_pad = e_aligned - e_len

  r_u16 = row_ids.astype(jnp.uint16)
  c_u16 = col_ids.astype(jnp.uint16)
  r_s_u16, c_s_u16 = jax.lax.sort(
      (r_u16, c_u16), dimension=0, is_stable=False, num_keys=1
  )
  r_local = jnp.bitwise_and(r_s_u16.astype(jnp.int32), b_dst - 1)
  packed_rc = jnp.bitwise_or(
      jnp.left_shift(r_local, 16),
      c_s_u16.astype(jnp.int32),
  )
  if e_pad > 0:
    packed_rc = jnp.pad(packed_rc, (0, e_pad), constant_values=0)

  num_dst_blocks = (num_rows + b_dst - 1) // b_dst
  thresh_ints = jnp.concatenate([
      (jnp.arange(num_dst_blocks + 1, dtype=jnp.int32) * b_dst).clip(
          0, num_rows
      ),
      jnp.array([max(0, num_rows - 1)], dtype=jnp.int32),
  ])
  thresh_u16 = thresh_ints.astype(jnp.uint16)
  raw_offsets = jnp.searchsorted(r_s_u16, thresh_u16, side='left').astype(
      jnp.int32
  )
  end_all = raw_offsets[num_dst_blocks]
  idx_last = raw_offsets[num_dst_blocks + 1]
  last_cnt = end_all - idx_last
  if num_rows == n_in and num_rows >= 64:
    safe_first = jnp.minimum(idx_last, e_len - 1)
    safe_last = jnp.maximum(0, end_all - 1)
    is_pad_self = (
        (last_cnt >= 4096)
        & (c_s_u16[safe_first].astype(jnp.int32) == (n_in - 1))
        & (c_s_u16[safe_last].astype(jnp.int32) == (n_in - 1))
    )
    eff_end = jnp.where(is_pad_self, idx_last, end_all)
    pad_self_cnt = jnp.where(is_pad_self, last_cnt, jnp.int32(0))
  else:
    eff_end = end_all
    pad_self_cnt = jnp.int32(0)
  offsets = raw_offsets[: num_dst_blocks + 1].at[num_dst_blocks].set(eff_end)
  offsets = jnp.concatenate([offsets, jnp.expand_dims(pad_self_cnt, 0)])
  offsets_len = max(32, ((num_dst_blocks + 2 + 31) // 32) * 32)
  offsets_pad = jnp.pad(offsets, (0, offsets_len - offsets.shape[0]))

  k_fn = _make_tiled_y_spmm_kernel(
      num_rows=num_rows,
      m_tile=m_tile,
      l_tile=l_tile,
      b_dst=b_dst,
      edge_blk=edge_blk,
  )
  x_3d = x_pad.reshape(n_in, m_tile, l_tile)
  y_3d = pl.pallas_call(
      k_fn,
      out_shape=jax.ShapeDtypeStruct((num_rows, m_tile, l_tile), jnp.float32),
      grid=(1,),
      in_specs=[
          pl.BlockSpec((offsets_len,), lambda _: (0,), memory_space=pltpu.SMEM),
          pl.BlockSpec(memory_space=pl.ANY),
          pl.BlockSpec(
              (n_in, m_tile, l_tile),
              lambda _: (0, 0, 0),
              pipeline_mode=pl.Buffered(buffer_count=1),
          ),
      ],
      out_specs=pl.BlockSpec(memory_space=pl.ANY),
      scratch_shapes=[
          pltpu.SMEM((edge_blk,), jnp.int32),
          pltpu.SMEM((1,), jnp.int32),
          pltpu.VMEM((b_dst + 4, m_tile, l_tile), jnp.float32),
          pltpu.SemaphoreType.DMA(()),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=('arbitrary',),
          vmem_limit_bytes=profile.vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
  )(offsets_pad, packed_rc, x_3d)
  y_2d = y_3d.reshape(num_rows, d_pad)
  if feat_dim < d_pad:
    y_2d = y_2d[:, :feat_dim]
  if orig_dtype != jnp.float32:
    y_2d = y_2d.astype(orig_dtype)
  return y_2d


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _spmm_tiled_y_unweighted(
    x_2d: jax.Array,
    row_ids: jax.Array,
    col_ids: jax.Array,
    num_rows: int,
    profile: TPUDeviceProfile,
) -> jax.Array:
  """Single-sort Tiled-Y 2D SpMM with custom VJP."""
  return _run_spmm_tiled_y_2d(x_2d, row_ids, col_ids, num_rows, profile)


def _spmm_tiled_y_unweighted_fwd(
    x_2d: jax.Array,
    row_ids: jax.Array,
    col_ids: jax.Array,
    num_rows: int,
    profile: TPUDeviceProfile,
):
  """Forward pass for `_spmm_tiled_y_unweighted`."""
  y = _run_spmm_tiled_y_2d(x_2d, row_ids, col_ids, num_rows, profile)
  return y, (row_ids, col_ids, x_2d.shape[0])


def _spmm_tiled_y_unweighted_bwd(
    num_rows: int,
    profile: TPUDeviceProfile,
    res,
    dy: jax.Array,
):
  """Backward pass for `_spmm_tiled_y_unweighted`."""
  del num_rows
  row_ids, col_ids, n_in = res
  dx = _run_spmm_tiled_y_2d(dy, col_ids, row_ids, n_in, profile)
  return dx, None, None


_spmm_tiled_y_unweighted.defvjp(
    _spmm_tiled_y_unweighted_fwd, _spmm_tiled_y_unweighted_bwd
)


def pallas_spmm(
    x: jax.Array,
    row_ids: jax.Array,
    col_ids: jax.Array,
    num_rows: int,
    values: jax.Array | None = None,
    *,
    schedule: tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]] | None = None,
    edge_block_size: int = _DEFAULT_EDGE_BLOCK_SIZE,
    unroll: int = _DEFAULT_UNROLL,
    interpret: bool = False,
    diag_weight: float = 0.0,
    device_profile: TPUDeviceProfile | str | None = None,
) -> jax.Array:
  """Computes Sparse-Dense Matrix Multiplication Y = A @ X in O(N) memory.

  For each edge e in 0..E-1:
    Y[row_ids[e]] += (values[e] if values is not None else 1) * X[col_ids[e]]
  plus optional fused self-loop `diag_weight * X` when `diag_weight != 0.0`.

  Args:
    x: Dense node features tensor of shape [num_cols, ...], e.g. [N, D] or
      [N, F, H].
    row_ids: 1D int32 array of destination (receiver) node indices of shape [E].
    col_ids: 1D int32 array of source (sender) node indices of shape [E].
    num_rows: Number of output rows (receiver nodes).
    values: Optional 1D array of edge weights of shape [E].
    schedule: Optional `(sched_fwd, sched_bwd)` from `prepare_spmm_schedule`.
    edge_block_size: Number of edges streamed into SMEM per grid step.
    unroll: Inner loop unroll factor with software store-to-load forwarding.
    interpret: If True, executes Pallas kernel in interpret mode (useful for
      CPU testing of the Pallas kernel).
    diag_weight: Optional scalar self-loop weight (`A + diag_weight * I`).
    device_profile: Optional `TPUDeviceProfile` or profile key (`'glp'`,
      `'vlp'`, `'gfl'`, `'gf'`, `'vf'`, `'pf'`).
      Auto-detected from the active TPU device when `None`.

  Returns:
    Output tensor of shape [num_rows, *x.shape[1:]] and same dtype as x.
  """
  orig_shape = x.shape
  n_in = orig_shape[0]
  trailing_shape = orig_shape[1:]
  feat_dim = math.prod(trailing_shape) if trailing_shape else 1
  x_2d = x.reshape(n_in, feat_dim)
  diag_w = float(diag_weight)

  if values is not None:
    y_2d = _pallas_spmm_weighted(
        x_2d,
        row_ids,
        col_ids,
        values,
        num_rows,
        edge_block_size,
        unroll,
        interpret,
        diag_w,
        device_profile,
    )
    return y_2d.reshape((num_rows, *trailing_shape))

  if jax.default_backend() != 'tpu' and not interpret:
    y_2d = _pallas_spmm_unweighted(
        x_2d,
        row_ids,
        col_ids,
        num_rows,
        edge_block_size,
        unroll,
        interpret,
        diag_w,
        device_profile,
    )
    return y_2d.reshape((num_rows, *trailing_shape))

  profile = get_tpu_device_profile(device_profile)
  max_working_set = _effective_max_working_set_bytes(profile)

  itemsize = jnp.dtype(x_2d.dtype).itemsize
  vreg_bytes_per_col = (n_in + int(num_rows)) * itemsize
  f32_bytes_per_col = (n_in + int(num_rows)) * 4
  _, _, d_pad = pick_3d_tile_shape(feat_dim, x_2d.dtype)

  min_1d_tile_bytes = min(
      vreg_bytes_per_col * min(d_pad, 256),
      f32_bytes_per_col * min(d_pad, 128),
  )
  if min_1d_tile_bytes > max_working_set or (
      f32_bytes_per_col * min(d_pad, 256) > max_working_set
      and d_pad >= 1024
  ):
    large_blk = (
        _DEFAULT_LARGE_EDGE_BLOCK_SIZE
        if edge_block_size == _DEFAULT_EDGE_BLOCK_SIZE and not interpret
        else edge_block_size
    )
    y_2d = _spmm_large_unweighted(
        x_2d,
        row_ids,
        col_ids,
        int(num_rows),
        large_blk,
        unroll,
        interpret,
        profile,
    )
    if diag_w != 0.0 and int(num_rows) == n_in:
      y_2d = y_2d + x_2d * jnp.asarray(diag_w, dtype=x_2d.dtype)
    return y_2d.reshape((num_rows, *trailing_shape))

  if schedule is None and (
      edge_block_size != _DEFAULT_EDGE_BLOCK_SIZE
      or unroll != _DEFAULT_UNROLL
  ):
    y_2d = _pallas_spmm_unweighted(
        x_2d,
        row_ids,
        col_ids,
        int(num_rows),
        edge_block_size,
        unroll,
        interpret,
        diag_w,
        profile,
    )
    return y_2d.reshape((num_rows, *trailing_shape))

  if (
      schedule is None
      and not interpret
      and row_ids.shape[0] >= 524288
      and n_in <= 65535
      and int(num_rows) <= 65535
      and f32_bytes_per_col * d_pad > max_working_set
      and max(n_in, int(num_rows)) * d_pad * itemsize + (2048 + 4) * d_pad * 4
      <= profile.vmem_limit_bytes - 1024 * 1024
  ):
    y_2d = _spmm_tiled_y_unweighted(
        x_2d,
        row_ids,
        col_ids,
        int(num_rows),
        profile,
    )
    if diag_w != 0.0 and int(num_rows) == n_in:
      y_2d = y_2d + x_2d * jnp.asarray(diag_w, dtype=x_2d.dtype)
    return y_2d.reshape((num_rows, *trailing_shape))

  b_size = (
      _INTERPRET_COSORT_BLOCK_SIZE if interpret else _DEFAULT_COSORT_BLOCK_SIZE
  )
  u_size = _INTERPRET_COSORT_UNROLL if interpret else _DEFAULT_COSORT_UNROLL

  if (
      schedule is not None
      and isinstance(schedule, tuple)
      and len(schedule) == 2
  ):
    sched_fwd, sched_bwd = schedule
    eff_b = (
        b_size
        if sched_fwd[0].shape[0] % b_size == 0
        else sched_fwd[0].shape[0]
    )
    y_2d = _pallas_spmm_scheduled(
        x_2d,
        sched_fwd,
        sched_bwd,
        int(num_rows),
        int(n_in),
        eff_b,
        u_size,
        interpret,
        diag_w,
        profile,
    )
  else:
    y_2d = _pallas_spmm_cosort_lazy(
        x_2d,
        row_ids,
        col_ids,
        int(num_rows),
        b_size,
        u_size,
        interpret,
        diag_w,
        profile,
    )
  return y_2d.reshape((num_rows, *trailing_shape))

