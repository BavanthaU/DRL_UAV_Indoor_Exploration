cmake_minimum_required(VERSION 3.18)
project(grid_planning_ext LANGUAGES CXX)

find_package(pybind11 REQUIRED)

pybind11_add_module(
    grid_planning_ext
    bindings.cpp
    src/astar.cpp
    src/frontier.cpp
    src/coverage.cpp
    src/connected_components.cpp
)
target_include_directories(grid_planning_ext PRIVATE include)
target_compile_features(grid_planning_ext PRIVATE cxx_std_17)
