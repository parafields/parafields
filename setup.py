from skbuild import setup

import pybind11

cmake_args = [
    f"-DCMAKE_PREFIX_PATH={pybind11.get_cmake_dir()}",
    "-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON",
    "-DBUILD_SINGLE_PRECISION=ON",
    "-DBUILD_DOUBLE_PRECISION=ON",
]

try:
    import mpi4py

    cmake_args.append(f"-DMPI4PY_INCLUDE_DIR={mpi4py.get_include()}")
except ImportError:
    pass

setup(
    packages=["parafields"],
    package_dir={"": "src"},
    include_package_data=True,
    package_data={"parafields": ["*.json"]},
    zip_safe=False,
    cmake_args=cmake_args,
    cmake_install_dir="src/parafields",
)
