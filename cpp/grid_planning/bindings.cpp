#include "grid_planning.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(grid_planning_ext, m) {
    m.doc() = "Grid planning acceleration for UAV exploration";
    m.def("astar_grid", &astar_grid);
    m.def("astar_to_any_goal", &astar_to_any_goal);
    m.def("extract_frontiers", &extract_frontiers);
    m.def("cluster_frontiers", &cluster_frontiers);
    m.def("update_coverage_bitset", &update_coverage_bitset);
    m.def("compute_coverage_ratio", &compute_coverage_ratio);
    m.def("connected_components", &connected_components);
}

