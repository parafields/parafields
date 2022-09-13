from skbuild import setup

import os
import pybind11


setup(
    packages=["parafields"],
    package_dir={"": "src"},
    include_package_data=True,
    package_data={"parafields": ["*.json"]},
    zip_safe=False,
    cmake_args=[
        f"-DCMAKE_PREFIX_PATH={os.path.dirname(pybind11.__file__)}",
        "-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON",
    ],
    cmake_install_dir="src/parafields",
)
