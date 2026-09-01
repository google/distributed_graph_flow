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

"""Scoped, hierarchical options and configuration context managers."""

from collections.abc import Iterator, Mapping, MutableMapping
import contextlib
import contextvars
from typing import Any

_SENTINEL = object()


class Context(Mapping[str, Any]):
  """Immutable context scope holding a dictionary of options and an optional parent scope."""

  def __init__(
      self,
      options: Mapping[str, Any] | None = None,
      parent: "Context | None" = None,
  ):
    self._options: dict[str, Any] = dict(options) if options is not None else {}
    self._parent = parent

  def get_option(self, key: str, default: Any = None) -> Any:
    """Gets the option value from this context or its parent scopes."""
    if key in self._options:
      return self._options[key]
    if self._parent is not None:
      return self._parent.get_option(key, default)
    return default

  def get(self, key: str, default: Any = None) -> Any:
    """Gets the option value from this context or its parent scopes."""
    return self.get_option(key, default)

  def __getitem__(self, key: str) -> Any:
    val = self.get_option(key, default=_SENTINEL)
    if val is _SENTINEL:
      raise KeyError(key)
    return val

  def __contains__(self, key: object) -> bool:
    if key in self._options:
      return True
    return self._parent is not None and key in self._parent

  def __iter__(self) -> Iterator[str]:
    return iter(self.to_dict())

  def __len__(self) -> int:
    return len(self.to_dict())

  def with_option(self, key: str, value: Any) -> "Context":
    """Returns a new Context with the option set (Copy-on-Write)."""
    new_options = self._options.copy()
    new_options[key] = value
    return Context(new_options, parent=self._parent)

  def with_options(self, **kwargs: Any) -> "Context":
    """Returns a new Context with the given options updated (Copy-on-Write)."""
    new_options = self._options.copy()
    new_options.update(kwargs)
    return Context(new_options, parent=self._parent)

  def without_option(self, key: str | None = None) -> "Context":
    """Returns a new Context with the option(s) reset (Copy-on-Write)."""
    if key is None:
      new_options = {}
    else:
      new_options = self._options.copy()
      new_options.pop(key, None)
    return Context(new_options, parent=self._parent)

  def to_dict(self) -> dict[str, Any]:
    """Returns the merged options dictionary up to this context level."""
    merged = self._parent.to_dict() if self._parent is not None else {}
    merged.update(self._options)
    return merged


class Manager(MutableMapping[str, Any]):
  """Thread-safe and coroutine-safe options manager for a subsystem."""

  def __init__(
      self,
      name: str,
      defaults: Mapping[str, Any] | None = None,
  ):
    self._default_options = Context(defaults)
    self._current_context: contextvars.ContextVar[Context] = (
        contextvars.ContextVar(name, default=self._default_options)
    )

  @property
  def default_options(self) -> Context:
    """Returns the base/default options context."""
    return self._default_options

  def get(self, key: str, default: Any = None) -> Any:
    """Gets the active option value from the current context stack."""
    return self._current_context.get().get_option(key, default)

  def set(self, key: str, value: Any) -> None:
    """Sets an option in the active context scope using Copy-on-Write."""
    current_context = self._current_context.get()
    self._current_context.set(current_context.with_option(key, value))

  def reset(self, key: str | None = None) -> None:
    """Resets an option or all options in the active context scope using Copy-on-Write."""
    self._current_context.set(self._current_context.get().without_option(key))

  def __getitem__(self, key: str) -> Any:
    return self._current_context.get()[key]

  def __setitem__(self, key: str, value: Any) -> None:
    self.set(key, value)

  def __delitem__(self, key: str) -> None:
    if key not in self:
      raise KeyError(key)
    self.reset(key)

  def __contains__(self, key: object) -> bool:
    return key in self._current_context.get()

  def __iter__(self) -> Iterator[str]:
    return iter(self._current_context.get())

  def __len__(self) -> int:
    return len(self._current_context.get())

  def to_dict(self) -> dict[str, Any]:
    """Returns the merged dictionary of active options."""
    return self._current_context.get().to_dict()

  @contextlib.contextmanager
  def context(self, **kwargs: Any) -> Iterator[Context]:
    """Context manager to temporarily override options in a stack-safe scope."""
    ctx = Context(kwargs.copy(), parent=self._current_context.get())
    token = self._current_context.set(ctx)
    try:
      yield ctx
    finally:
      self._current_context.reset(token)

  def __call__(self, **kwargs: Any) -> Any:
    """Allows manager instance to be used directly as a context manager: `with options(...)`."""
    return self.context(**kwargs)
