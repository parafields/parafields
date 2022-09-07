from skbuild import setup

import os
import pybind11


setup(
    packages=["parafields"],
    package_dir={"": "src"},
    zip_safe=False,
    cmake_args=[
        f"-DCMAKE_PREFIX_PATH={os.path.dirname(pybind11.__file__)}",
    ],
    cmake_install_dir="src/parafields",
)
