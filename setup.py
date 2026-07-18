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

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install
from setuptools.dist import Distribution


class InstallPlatlib(install):

  def finalize_options(self):
    install.finalize_options(self)
    if self.distribution.has_ext_modules():
      self.install_lib = self.install_platlib


class BinaryDistribution(Distribution):

  def has_ext_modules(self):
    return True

  def is_pure(self):
    return False


setup(
    name="dgf",
    version="0.0.4",
    author="Mathieu Guillame-Bert, Brandon Mayer",
    description="Distributed Graph Flow",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/google/dgf",
    project_urls={
        "Source": "https://github.com/google/dgf.git",
        "Tracker": "https://github.com/google/dgf/issues",
    },
    entry_points={
        "console_scripts": [
            "dgf-validate-graph=dgf.src.bin:validate_graph",
        ],
    },
    license="Apache 2.0",
    distclass=BinaryDistribution,
    include_package_data=True,
    zip_safe=False,
    packages=find_packages(),
    package_data={"dgf": ["LICENSE", "**/*.so"]},
    cmdclass={"install": InstallPlatlib},
    python_requires=">=3.11",
    install_requires=[str(r) for r in open("requirements.txt").readlines()],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
