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

"""Monte-Carlo verification checks for variance inflation and empirical moments.

Simulations S1-S7 verify:
  S1: Sturmian count autocorrelation in equal-count windows.
  S2: High-accuracy count component and lag-1 tests.
  S3: Bursty time buckets with unequal counts.
  S4: Cluster graph edge statistic and local VIF bounds.
  S5: Correlated R^2 bounds without pairing.
  S6: Uncorrelated but dependent totals (xi_c * xi_{c-1}).
  S7: Edge statistics on hub-heavy cluster graphs.
"""

from collections.abc import Sequence
import json
import math
import os
from typing import Any

from absl import app
from absl import flags
import numpy as np

DEFAULT_SEED = 42

_MODE = flags.DEFINE_enum(
    "mode",
    "quick",
    ["quick", "full"],
    "Execution mode: 'quick' or 'full'.",
)
_SEED = flags.DEFINE_integer(
    "seed",
    DEFAULT_SEED,
    "Random seed base.",
)
_OUTPUT = flags.DEFINE_string(
    "output",
    None,
    "Optional output path for JSONL results.",
)

_Z95 = 1.6448536269514722


def _lag1(x: np.ndarray, center: bool) -> np.ndarray:
  """Row-wise lag-1 autocorrelation (numerator over G-1 pairs)."""
  if center:
    x = x - x.mean(axis=1, keepdims=True)
  num = np.sum(x[:, :-1] * x[:, 1:], axis=1)
  den = np.sum(x * x, axis=1)
  return num / den


def _fisher_reject(r1: np.ndarray, g: int) -> np.ndarray:
  """One-sided Fisher-z lag-1 test as in refutation.check_uncorrelated."""
  r = np.clip(r1, -0.9999, 0.9999)
  return np.arctanh(r) * math.sqrt(g - 4) > _Z95


def _edge_stat(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
  """Self-normalised edge statistic; rows of x are replications."""
  prod = x[:, edges[:, 0]] * x[:, edges[:, 1]]
  return prod.sum(axis=1) / np.sqrt(np.sum(prod * prod, axis=1))


def _residuals(s: np.ndarray, counts: np.ndarray) -> np.ndarray:
  """S_c - n_c * theta_hat, row-wise."""
  theta = s.sum(axis=1, keepdims=True) / counts.sum()
  return s - counts[None, :] * theta


def _ar1(rng: np.random.Generator, reps: int, g: int, rho: float) -> np.ndarray:
  eta = np.empty((reps, g))
  eta[:, 0] = rng.normal(size=reps)
  scale = math.sqrt(1.0 - rho * rho)
  for t in range(1, g):
    eta[:, t] = rho * eta[:, t - 1] + scale * rng.normal(size=reps)
  return eta


def _window_counts(n: int, g: int) -> np.ndarray:
  return np.bincount((np.arange(n) * g) // n, minlength=g).astype(float)


def run_s1() -> list[dict[str, Any]]:
  """S1 equal-count windows: lag-1 autocorrelation of the counts."""
  print("\nS1 equal-count windows: lag-1 autocorrelation of the counts")
  print("   n      G    frac   rho_delta  predicted -f/(1-f)")
  records = []
  for n, g in [(20070, 200), (20050, 200), (20020, 200), (20180, 200),
               (20199, 200), (100333, 1000), (12345, 100), (50017, 100)]:
    counts = _window_counts(n, g)
    d = counts - counts.mean()
    rho = float(np.sum(d[:-1] * d[1:]) / np.sum(d * d))
    frac = n / g - math.floor(n / g)
    f = min(frac, 1.0 - frac)
    pred = -f / (1.0 - f)
    print(f"{n:7d} {g:5d} {frac:6.3f} {rho:10.4f} {pred:10.4f}")
    records.append({
        "sim": "S1",
        "n": n,
        "g": g,
        "frac": frac,
        "rho": rho,
        "pred": pred,
    })
  return records


def run_s2(reps: int = 4000, seed: int = 1) -> list[dict[str, Any]]:
  """S2 lag-1 test, equal-count windows."""
  print(f"\nS2 lag-1 test, equal-count windows n=20070 G=200, {reps} reps")
  print("theta  b(AR)  rho_tilde_mean  rej_raw  rej_resid  rej_sn")
  rng = np.random.default_rng(seed)
  n, g = 20070, 200
  counts = _window_counts(n, g)
  records = []
  thetas = (0.9, 0.99, 0.999) if reps >= 1000 else (0.9, 0.99)
  bs = (0.0, 0.3, 0.6) if reps >= 1000 else (0.0, 0.3)
  for theta in thetas:
    for b in bs:
      if b > 0:
        eta = _ar1(rng, reps, g, 0.5)
        err = np.minimum((1.0 - theta) * np.exp(b * eta - b * b / 2), 1.0)
      else:
        err = np.full((reps, g), 1.0 - theta)
      s = rng.binomial(counts[None, :].astype(int), 1.0 - err).astype(float)
      e = _residuals(s, counts)
      th = s.sum(axis=1) / counts.sum()
      dd = counts - counts.mean()
      rho_t = th * th * np.sum(dd * dd) / np.sum(e * e, axis=1)
      raw = float(_fisher_reject(_lag1(s, True), g).mean())
      res = float(_fisher_reject(_lag1(e, False), g).mean())
      edges = np.stack([np.arange(g - 1), np.arange(1, g)], axis=1)
      sn = float(np.mean(_edge_stat(e, edges) > _Z95))
      print(f"{theta:5.3f} {b:5.2f} {float(np.mean(rho_t)):14.4f}"
            f" {raw:8.4f} {res:9.4f} {sn:7.4f}")
      records.append({
          "sim": "S2",
          "theta": theta,
          "b": b,
          "rho_tilde_mean": float(np.mean(rho_t)),
          "rej_raw": raw,
          "rej_resid": res,
          "rej_sn": sn,
      })
  return records


def run_s3(reps: int = 4000, seed: int = 2) -> list[dict[str, Any]]:
  """S3 time buckets with unequal counts."""
  print(f"\nS3 time buckets with unequal counts, G=300, theta=0.9, {reps} reps")
  print("counts   b(AR)  CV_n^2  fz_raw  fz_resid  fz_norm  sn_resid  sn_norm")
  rng = np.random.default_rng(seed)
  g, theta = 300, 0.9
  edges = np.stack([np.arange(g - 1), np.arange(1, g)], axis=1)
  lognormal = np.maximum(
      1.0, np.round(np.exp(rng.normal(math.log(100.0), 1.0, g)))
  )
  trend = np.round(100.0 * np.exp(np.linspace(-1.0, 1.0, g)))
  targets = [("lognorm", lognormal), ("trend", trend)] if reps >= 1000 else [("lognorm", lognormal)]
  records = []
  for name, counts in targets:
    cv2 = float(g * np.sum(counts**2) / np.sum(counts)**2 - 1.0)
    for b in (0.0, 0.3):
      if b > 0:
        eta = _ar1(rng, reps, g, 0.5)
        err = np.minimum((1.0 - theta) * np.exp(b * eta - b * b / 2), 1.0)
      else:
        err = np.full((reps, g), 1.0 - theta)
      s = rng.binomial(counts[None, :].astype(int), 1.0 - err).astype(float)
      e = _residuals(s, counts)
      z = e / np.sqrt(counts)[None, :]
      fz_raw = float(_fisher_reject(_lag1(s, True), g).mean())
      fz_res = float(_fisher_reject(_lag1(e, False), g).mean())
      fz_norm = float(_fisher_reject(_lag1(z, True), g).mean())
      sn_res = float(np.mean(_edge_stat(e, edges) > _Z95))
      sn_norm = float(np.mean(_edge_stat(z, edges) > _Z95))
      print(f"{name:8s} {b:5.2f} {cv2:7.3f} {fz_raw:7.4f} {fz_res:9.4f}"
            f" {fz_norm:8.4f} {sn_res:9.4f} {sn_norm:8.4f}")
      records.append({
          "sim": "S3",
          "name": name,
          "b": b,
          "cv2": cv2,
          "fz_raw": fz_raw,
          "fz_res": fz_res,
          "fz_norm": fz_norm,
          "sn_res": sn_res,
          "sn_norm": sn_norm,
      })
  return records


def run_s4(reps: int = 4000, seed: int = 3) -> list[dict[str, Any]]:
  """S4 cluster graph (random geometric) heteroscedastic Gaussian totals."""
  g = 500 if reps >= 1000 else 100
  print(f"\nS4 cluster graph (random geometric, G={g}), {reps} reps")
  rng = np.random.default_rng(seed)
  pts = rng.random((g, 2))
  radius = math.sqrt(6.0 / (math.pi * g))
  dist2 = np.sum((pts[:, None, :] - pts[None, :, :])**2, axis=-1)
  adj = (dist2 < radius * radius) & ~np.eye(g, dtype=bool)
  a = adj.astype(float)
  edges = np.argwhere(np.triu(adj, 1))
  deg = a.sum(axis=1)
  lam = float(np.max(np.linalg.eigvalsh(a)))
  n_e = edges.shape[0]
  print(f"|E|={n_e} mean_deg={float(deg.mean()):.2f}"
        f" d_max={int(deg.max())} lambda_max={lam:.2f}")
  counts = np.maximum(
      1.0, np.round(np.exp(rng.normal(math.log(100.0), 0.7, g)))
  )
  d_half = np.sqrt(counts)
  print("b     VIF_true  VIF_Wlocal  kappaW_mean  rej_sn_resid  rej_sn_norm")
  bs = (0.0, 0.03, 0.06, 0.1) if reps >= 1000 else (0.0, 0.06)
  records = []
  for b in bs:
    root = np.eye(g) + b * a
    cov = (d_half[:, None] * (root @ root)) * d_half[None, :]
    vif = float(cov.sum() / np.trace(cov))
    vif_w = float(1.0 + np.sum(cov * a) / np.trace(cov))
    xi = rng.normal(size=(reps, g))
    x = (xi @ root) * d_half[None, :]
    s = counts[None, :] * 0.3 + x
    e = _residuals(s, counts)
    z = e / d_half[None, :]
    kappa_w = 1.0 + 2.0 * np.sum(
        e[:, edges[:, 0]] * e[:, edges[:, 1]], axis=1
    ) / np.sum(e * e, axis=1)
    rej_e = float(np.mean(_edge_stat(e, edges) > _Z95))
    rej_z = float(np.mean(_edge_stat(z, edges) > _Z95))
    print(f"{b:4.2f} {vif:9.3f} {vif_w:11.3f} {float(np.mean(kappa_w)):12.3f}"
          f" {rej_e:13.4f} {rej_z:12.4f}")
    records.append({
        "sim": "S4",
        "b": b,
        "vif": vif,
        "vif_w": vif_w,
        "kappa_w": float(np.mean(kappa_w)),
        "rej_e": rej_e,
        "rej_z": rej_z,
    })
  return records


def run_s5(reps: int = 2000, seed: int = 4) -> list[dict[str, Any]]:
  """S5 correlated R^2 (no pairing)."""
  print(f"\nS5 correlated R^2 (no pairing), theta_A=0.2, Var(y_I)=1, {reps} reps")
  print("   G   m icc_y icc_e  E[Vhat]  1-Var(ybar)  cover  miss_lo"
        "  miss_hi  med_lo  med_hi  emp_R2")
  rng = np.random.default_rng(seed)
  alpha, m_item, kappa = 0.05, 3.0, 1.0
  grid = (
      [(4000, 25, 0.5, 0.5), (1000, 25, 0.9, 0.5),
       (4000, 5, 0.5, 0.9), (20000, 5, 0.2, 0.2)]
      if reps >= 1000
      else [(1000, 25, 0.5, 0.5)]
  )
  records = []
  for g, m, icc_y, icc_e in grid:
    n = g * m

    def cluster_sums(total_var: float, icc: float):
      u = rng.normal(0.0, math.sqrt(total_var * icc), (reps, g))
      sv = rng.normal(0.0, math.sqrt(total_var * (1.0 - icc) * m), (reps, g))
      chi = rng.chisquare(m - 1, (reps, g))
      s1 = m * u + sv
      s2 = m * u * u + 2.0 * u * sv + sv * sv / m + total_var * (
          1.0 - icc
      ) * chi
      return s1, s2

    sy, sy2 = cluster_sums(1.0, icc_y)
    _, se2 = cluster_sums(0.2, icc_e)
    ybar = sy.sum(axis=1) / n
    vhat = sy2.sum(axis=1) / n - ybar**2
    mse = se2.sum(axis=1) / n
    var_ybar = icc_y / g + (1.0 - icc_y) / n
    m_c = 1.0 + (m_item - 1.0)
    a1 = alpha / 4.0
    c1 = math.sqrt((m_c - 1.0) * kappa * (1.0 - a1) / (a1 * g))
    a2 = alpha / 8.0
    c2 = math.sqrt((m_c - 1.0) * kappa * (1.0 - a2) / (a2 * g))
    cheb = kappa * m / (n * a2)
    l_b = vhat / (1.0 + c1)
    u_a = mse / (1.0 - c1)
    l_a = mse / (1.0 + c1)
    den = 1.0 - c2 - cheb
    u_b = vhat / den if den > 0 else np.full_like(vhat, np.inf)
    lo = 1.0 - u_a / l_b
    hi = 1.0 - l_a / u_b
    r2 = 0.8
    miss_lo = float(np.mean(lo > r2))
    miss_hi = float(np.mean(hi < r2))
    cover = 1.0 - miss_lo - miss_hi
    emp = float(np.mean(1.0 - mse / vhat))
    print(f"{g:5d} {m:3d} {icc_y:5.2f} {icc_e:5.2f} {float(np.mean(vhat)):8.5f}"
          f" {1.0 - var_ybar:12.5f} {cover:6.4f}"
          f" {miss_lo:8.4f} {miss_hi:8.4f} {float(np.median(lo)):7.4f}"
          f" {float(np.median(hi)):7.4f} {emp:7.4f}")
    records.append({
        "sim": "S5",
        "g": g,
        "m": m,
        "icc_y": icc_y,
        "icc_e": icc_e,
        "cover": cover,
        "miss_lo": miss_lo,
        "miss_hi": miss_hi,
        "med_lo": float(np.median(lo)),
        "med_hi": float(np.median(hi)),
        "emp_r2": emp,
    })
  return records


def run_s6(reps: int = 4000, seed: int = 5) -> list[dict[str, Any]]:
  """S6 totals eps_c = xi_c * xi_{c-1} (uncorrelated, dependent)."""
  print(f"\nS6 totals eps_c = xi_c * xi_{{c-1}} (uncorrelated, dependent), {reps} reps")
  print("   G  fisher_z  self_norm")
  rng = np.random.default_rng(seed)
  gs = (100, 400, 1600) if reps >= 1000 else (100,)
  records = []
  for g in gs:
    xi = rng.normal(size=(reps, g + 1))
    eps = xi[:, 1:] * xi[:, :-1]
    edges = np.stack([np.arange(g - 1), np.arange(1, g)], axis=1)
    fz = float(_fisher_reject(_lag1(eps, True), g).mean())
    centred = eps - eps.mean(axis=1, keepdims=True)
    sn = float(np.mean(_edge_stat(centred, edges) > _Z95))
    print(f"{g:5d} {fz:9.4f} {sn:10.4f}")
    records.append({
        "sim": "S6",
        "g": g,
        "fz": fz,
        "sn": sn,
    })
  return records


def run_s7(reps: int = 4000, seed: int = 6) -> list[dict[str, Any]]:
  """S7 edge statistic on hub-heavy cluster graphs."""
  print(f"\nS7 edge statistic on hub-heavy cluster graphs, {reps} reps")
  print("graph           G   |E|  d_max  lam^2/2E  dmax^2/2E  rej_sn")
  rng = np.random.default_rng(seed)

  def barabasi_albert(g: int, m: int) -> np.ndarray:
    adj = np.zeros((g, g), dtype=bool)
    targets = list(range(m))
    repeated: list[int] = []
    for v in range(m, g):
      for t in set(targets):
        adj[v, t] = adj[t, v] = True
      repeated.extend(targets)
      repeated.extend([v] * m)
      picks = rng.choice(len(repeated), size=m * 3)
      targets = list(dict.fromkeys(repeated[int(p)] for p in picks))[:m]
    return adj

  def hubs(g: int, n_hubs: int, p_hub: float, deg: int) -> np.ndarray:
    adj = np.zeros((g, g), dtype=bool)
    for v in range(g):
      for w in rng.choice(g, size=deg, replace=False):
        if w != v:
          adj[v, w] = adj[w, v] = True
    for h in range(n_hubs):
      mask = rng.random(g) < p_hub
      mask[h] = False
      adj[h, mask] = True
      adj[mask, h] = True
    return adj

  graphs = (
      [
          ("BA m=2", barabasi_albert(500, 2)),
          ("BA m=2", barabasi_albert(2000, 2)),
          ("hubs 1x0.5", hubs(500, 1, 0.5, 2)),
          ("hubs 3x0.9", hubs(500, 3, 0.9, 2)),
          ("hubs 1x0.9", hubs(2000, 1, 0.9, 2)),
      ]
      if reps >= 1000
      else [("BA m=2", barabasi_albert(100, 2))]
  )
  records = []
  for name, adj in graphs:
    g = adj.shape[0]
    a = adj.astype(float)
    edges = np.argwhere(np.triu(adj, 1))
    n_e = edges.shape[0]
    deg = a.sum(axis=1)
    lam = float(np.max(np.linalg.eigvalsh(a)))
    x = rng.normal(size=(reps, g))
    x = x - x.mean(axis=1, keepdims=True)
    rej = float(np.mean(_edge_stat(x, edges) > _Z95))
    print(f"{name:12s} {g:5d} {n_e:5d} {int(deg.max()):6d}"
          f" {lam * lam / (2 * n_e):9.4f}"
          f" {float(deg.max())**2 / (2 * n_e):10.4f} {rej:7.4f}")
    records.append({
        "sim": "S7",
        "name": name,
        "g": g,
        "n_e": n_e,
        "rej": rej,
    })
  return records


def run_all(mode: str = "quick", seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
  """Runs all simulations S1-S7."""
  reps = 100 if mode == "quick" else 4000
  s5_reps = 100 if mode == "quick" else 2000

  all_records: list[dict[str, Any]] = []
  all_records.extend(run_s1())
  all_records.extend(run_s2(reps=reps, seed=seed + 1))
  all_records.extend(run_s3(reps=reps, seed=seed + 2))
  all_records.extend(run_s4(reps=reps, seed=seed + 3))
  all_records.extend(run_s5(reps=s5_reps, seed=seed + 4))
  all_records.extend(run_s6(reps=reps, seed=seed + 5))
  all_records.extend(run_s7(reps=reps, seed=seed + 6))
  return all_records


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  records = run_all(mode=_MODE.value, seed=_SEED.value)

  if _OUTPUT.value:
    out_dir = os.path.dirname(_OUTPUT.value)
    if out_dir:
      os.makedirs(out_dir, exist_ok=True)
    with open(_OUTPUT.value, "w") as f:
      for r in records:
        f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(records)} records to {_OUTPUT.value}")


if __name__ == "__main__":
  app.run(main)
