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

"""Various utilities to cache data to accelerate/skip IO operations.

For example, instead of loading a dataset in a slow-to-read format
(e.g., HGraph), this utility gives the ability to create a fast-to-read
local cache.

Such caching is useful for iterative development code.

The following example shows how to load a dataset from CNS, and create a local
cache.

```python
# Load a hgraph in memory. If the cache already exist, use it (which is
# ~400x faster than loading the actual hgraph).
graph, schema = dgf.io.cache("/tmp/cache.pkl",
  lambda: dgf.io.read_graphai_hgraph("/cns/path/to/hgraph")
  )
```
"""

import inspect
import pickle
from typing import Callable, Optional, Sequence, TypeVar, Union
from dgf.src.util import filesystem as fs


def _load_from_pickle(path: str):
  with fs.open_read(path, binary=True) as f:
    return pickle.load(f)


def _save_to_pickle(data, path: str):
  with fs.open_write(path, binary=True) as f:
    pickle.dump(data, f)


T = TypeVar("T")


def cache(
    path: str,
    create_fn: Callable[..., T],
    variable_names: Optional[Union[str, Sequence[str]]] = None,
) -> T:
  """Returns and caches the variable(s) created by "create_fn".

  On the first call, "create_fn" is called, and its content saved in "path"
  using pickle. On the next calls, the variable is loaded from "path" instead.

  If "variables" is provided, and variable(s) with the same name exist, return
  them directly.

  Pickle is not a good storage format. Only use "cache" to temporarly cache
  data.

  Usage example:

  ```python
  # Load a hgraph in memory. If the cache already exist, use it (which is
  # ~400x faster than loading the actual hgraph).
  graph, schema = dgf.io.cache("/tmp/cache.pkl",
    lambda: dgf.io.read_graphai_hgraph("/cns/path/to/hgraph")
    )
  ```

  ```python
  # Same as before, but even more efficient in Colab when re-runing cells.
  graph, schema = dgf.io.cache("/tmp/cache.pkl",
    lambda: dgf.io.read_graphai_hgraph("/cns/path/to/hgraph"),
    variable_names=("graph","schema"),
    )
  ```

  Args:
    path: The file system path where the data will be cached.
    create_fn: A callable that produces the data to be cached. This function
      will only be called if the cached file does not exist or if
      `variable_names` are not found.
    variable_names: Optional. Either a single string or a sequence of strings
      representing variable names. If provided, the function will first check if
      variables with these names exist in the caller's local scope. If all
      specified variables are found, their values are returned directly,
      bypassing the cache.

  Returns:
    The cached data or the result of `create_fn()`. If `variable_names` was
    provided and matched multiple variables, a tuple of the variable values
    is returned.
  """
  if variable_names is not None:
    if isinstance(variable_names, str):
      return_tuple = False
      variable_names = (variable_names,)
    elif isinstance(variable_names, tuple):
      return_tuple = True
    else:
      raise TypeError(
          "`variable_names` must be a str, tuple, or None, but got "
          f"{type(variable_names).__name__}"
      )

    frame = inspect.currentframe()
    if frame is not None and frame.f_back is not None:
      caller_locals = frame.f_back.f_locals
      found_vars = []
      for name in variable_names:
        if name in caller_locals:
          found_vars.append(caller_locals[name])

      if len(found_vars) == len(variable_names):
        if return_tuple:
          return tuple(found_vars)  # pyrefly: ignore[bad-return]
        else:
          return found_vars[0]

  if fs.exists(path):
    return _load_from_pickle(path)

  data = create_fn()
  _save_to_pickle(data, path)

  return data
