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

"""Automatically creates the "api.md" page of the documentation.

Usage:

1.
In google3, run:
blaze run -c opt //third_party/py/dgf/script:gen_api

2.
Open the generated  "api.md" file and auto-format it.

3.
Create and submit CL.
"""

import dataclasses
import inspect
import os
import pprint
from typing import List, Optional
from absl import app
import dgf


# Where to save the markdown file with the api.
DST_API_MD = "third_party/py/dgf/g3doc/api.md"
DST_API2_MD = "third_party/py/dgf/doc/docs/api.md"


@dataclasses.dataclass
class ItemDef:
  """Function / class in the documenation."""

  name: str  # e.g. read_graph
  full_name: str  # e.g. dgf.io.read_graph
  module: str  # e.g. dgf.io
  source_file: str  # e.g. dgf/src/io/hgraph_in_memory.py
  line_number: int  # e.g. 132
  doc_first_line: Optional[
      str
  ]  # e.g. Reads a set of in memory graphs from disk stored.

  def doc_url(self):
    return (
        f"http://google3/third_party/py/{self.source_file};l={self.line_number}"
    )


@dataclasses.dataclass
class ModuleDef:
  """Module / sub-module in the documentation."""

  name: str  # e.g. io
  doc_first_line: str  # e.g. IO functions and classes
  items: List[ItemDef]
  submodules: List["ModuleDef"]


def extract_module(module, alias: List[str], depth: int = 0) -> ModuleDef:
  """Extracts the structure of the DGF API."""
  module_doc = get_first_doc_line(module)
  if module_doc is None:
    raise ValueError(f"API Module {'.'.join(alias)} has no docstring.")
  module_def = ModuleDef(
      name=".".join(alias),
      doc_first_line=module_doc,
      items=[],
      submodules=[],
  )
  if depth > 5:
    # To be safe.
    raise ValueError(
        f"Maximum recursion depth exceeded while extracting module: {alias}"
    )

  for name, obj in vars(module).items():
    if name.startswith("_"):
      # Skip hidden symbols
      continue
    if inspect.ismodule(obj):
      if not obj.__name__.startswith("dgf.src.api."):
        # Skip non DGF API symbols.
        continue
      module_def.submodules.append(
          extract_module(obj, alias + [name], depth=depth + 1)
      )

    elif inspect.isfunction(obj) or inspect.isclass(obj):
      try:
        source_file = inspect.getsourcefile(obj)
        _, line_number = inspect.getsourcelines(obj)
      except TypeError as e:
        print(f"\tCould not get source info: {e}")
        raise e

      module_def.items.append(
          ItemDef(
              name=name,
              full_name=".".join(alias) + "." + name,
              module=".".join(alias),
              doc_first_line=get_first_doc_line(obj),
              source_file=prune_source_file_path(source_file),
              line_number=line_number,
          )
      )
    else:
      pass

  module_def.items.sort(key=lambda item: item.full_name)
  # Sort submodules, ensuring 'dgf.beam' modules come after others.
  module_def.submodules.sort(
      key=lambda module: (
          1 if module.name.startswith("dgf.beam") else 0,
          module.name,
      )
  )

  return module_def


def prune_source_file_path(path: str) -> str:
  """Returns the part of the path after 'dgf/src'."""
  marker = "dgf/src/"
  _, dgf_path = path.split(marker)
  return f"{marker}{dgf_path}"


def get_first_doc_line(module) -> Optional[str]:
  """Gets the first non-empty line of a module's docstring."""
  doc = inspect.getdoc(module)
  if not doc:
    return None
  for line in doc.splitlines():
    if line.strip():
      return line.strip()
  return None


def write_md_file(global_module_def: ModuleDef, path: str, url: bool = True):
  print("Writing documentation to", path)
  with open(path, "w") as f:
    # Header
    f.write("""\
# (Distributed) GraphFlow API

All the Apache Beam distributed functions / classes are defined in `dgf.beam.*`. All the other (e.g., in-process, in-memory) functions and classes are defined in `dgf.*` directly:

""")

    if not url:
      f.write("""\

To see the documentation of a function, use the python build-in `help` method or `?` in a colab e.g., `?dgf.io.read_feature_statistics`.

""")

    # The top modules.
    for top_module in global_module_def.submodules:
      f.write(f"*   `{top_module.name}.*`: {top_module.doc_first_line}\n")

    f.write(
        "\n\nFunctions not yet part of the official API are available under"
        " `dgf.src.*`. Those are not listed in this page.\n\n"
    )

    def rec(module_def: ModuleDef, depth: int = 0):

      if depth > 0:
        # Note: The "dgf" header is written manually above.
        section = "#" * (depth + 2)
        f.write(f"{section} Module `{module_def.name}`\n\n")
        if module_def.doc_first_line is not None:
          f.write(f"{module_def.doc_first_line}\n\n")

      for item in module_def.items:
        if url:
          f.write(
              f"*   [{item.full_name}]({item.doc_url()}):"
              f" {item.doc_first_line}\n"
          )
        else:
          f.write(f"*   `{item.full_name}`: {item.doc_first_line}\n")
      f.write("\n\n")

      for sub_module in module_def.submodules:
        rec(sub_module, depth + 1)
      f.write("\n\n")

    rec(global_module_def)


def main(argv):
  del argv  # Unused.

  g3_dir = os.getenv("BUILD_WORKSPACE_DIRECTORY") or ""

  module_def = extract_module(dgf, ["dgf"])
  pprint.pprint(module_def)

  write_md_file(module_def, os.path.join(g3_dir, DST_API_MD))
  write_md_file(module_def, os.path.join(g3_dir, DST_API2_MD), url=False)


if __name__ == "__main__":
  app.run(main)
