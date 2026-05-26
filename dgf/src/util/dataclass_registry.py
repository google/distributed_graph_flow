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

"""Define an annotation to serialize/deserialize layer config."""

import dataclasses
from typing import Any, Dict, List, Optional
import dataclasses_json

# Name of the field containing the type / class name.
TYPE_FIELD = "__type"


def create_registry(name: str) -> "Registry":
  """Creates a new registry with the given name.

  Args:
    name: The name of the registry.

  Returns:
    A new registry.

  Example:
    ```python
    registry = dataclass_registry.create_registry("my_registry")

    @registry.register
    @dataclasses_json.dataclass_json
    @dataclasses.dataclass
    class A:
      x: int


    @registry.register
    @dataclasses_json.dataclass_json
    @dataclasses.dataclass
    class B:
      a: Any = registry.field()

     b = B(a=A(2))
    assert b == B.from_json(b.to_json())
    ```
  """
  return Registry(name)


class Registry:
  """A registry for classes that can be serialized and deserialized.

  This class is used to register classes that can be serialized and deserialized
  by the `dataclasses_json` library. It also provides a custom field type that
  can be used to serialize and deserialize objects of registered classes.
  """

  def __init__(self, name: str):
    self._name = name
    self._registered_classes: Dict[str, Any] = {}

  def register(self, cls):
    """Registers a class with the registry.

    Args:
      cls: The class to register.

    Returns:
      The registered class.
    """
    if not dataclasses.is_dataclass(cls):
      raise TypeError("Only dataclasses can be registered.")
    if not hasattr(cls, "from_dict") or not hasattr(cls, "to_dict"):
      raise TypeError("Only dataclasses_json classes can be registered.")

    key = self._get_key(cls)
    if key in self._registered_classes:
      raise ValueError(f"Class {key!r} already registered.")
    self._registered_classes[key] = cls
    return cls

  def field(
      self, default=dataclasses.MISSING, default_factory=dataclasses.MISSING
  ):
    """Returns a field for en-/decoding objects of registered classes.

    Args:
      default: The default value of the field.
      default_factory: A 0-argument function called to initialize a field's
        value.

    Returns:
      A dataclass field.
    """
    kwargs = {}
    # dataclasses.MISSING is a sentinel value used to detect if a default
    # or default_factory was provided by the user. We only add them to kwargs
    # if they are not dataclasses.MISSING.
    if default is not dataclasses.MISSING:
      kwargs["default"] = default
    if default_factory is not dataclasses.MISSING:
      kwargs["default_factory"] = default_factory

    return dataclasses.field(  # pytype: disable=wrong-keyword-args
        **kwargs,
        metadata=dataclasses_json.config(
            encoder=self._encode, decoder=self._decode
        ),
    )

  def field_list(self):
    """Returns a field for en-/decoding a list of objects of registered classes.

    Returns:
      A dataclass field.
    """
    return dataclasses.field(
        metadata=dataclasses_json.config(
            encoder=self._encode_list, decoder=self._decode_list
        ),
    )

  def _get_key(self, cls):
    return f"{self._name}.{cls.__name__}"

  def _encode(self, item: Any) -> Optional[Dict[str, Any]]:
    """Encodes an object of a registered class to a JSON dictionary."""
    if item is None:
      return None
    encoded_item = item.to_dict()
    if TYPE_FIELD in encoded_item:
      raise ValueError(f"'{TYPE_FIELD}' is a reserved field name.")
    key = self._get_key(item.__class__)
    if key not in self._registered_classes:
      raise ValueError(
          f"Unknown type: {key}. Available:"
          f" {list(self._registered_classes.keys())}"
      )
    encoded_item[TYPE_FIELD] = key
    return encoded_item

  def _decode(self, encoded_item: Dict[str, Any]) -> Any:
    """Decodes a JSON dictionary to an object of a registered class."""
    item_type = encoded_item.get(TYPE_FIELD)
    if not item_type:
      raise ValueError(f"Missing {TYPE_FIELD} in data: {encoded_item}")
    cls = self._registered_classes.get(item_type)
    if cls is None:
      raise ValueError(
          f"Unknown type: {item_type}. Available:"
          f" {list(self._registered_classes.keys())}"
      )
    data = encoded_item.copy()
    data.pop(TYPE_FIELD)
    return cls.from_dict(data)

  def _encode_list(self, items: List[Any]) -> Any:
    """Encodes a list of objects of registered classes."""
    if items is None:
      return None
    return [self._encode(item) for item in items]

  def _decode_list(self, encoded_items: Any) -> List[Any]:
    """Decodes a list of JSON dictionaries to objects of registered classes."""
    return [self._decode(item) for item in encoded_items]

  def registered_keys(self) -> List[str]:
    return sorted(list(self._registered_classes.keys()))
