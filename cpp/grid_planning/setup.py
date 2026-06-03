from __future__ import annotations

from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

sources = [
    "bindings.cpp",
    "src/astar.cpp",
    "src/frontier.cpp",
    "src/coverage.cpp",
    "src/connected_components.cpp",
]

setup(
    name="grid_planning_ext",
    version="0.1.0",
    ext_modules=[
        Pybind11Extension(
            "grid_planning_ext",
            [str(Path(source)) for source in sources],
            include_dirs=["include"],
            cxx_std=17,
        )
    ],
    cmdclass={"build_ext": build_ext},
)

