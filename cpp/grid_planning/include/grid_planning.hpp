#pragma once

#include <pybind11/pybind11.h>

pybind11::tuple astar_grid(pybind11::tuple start, pybind11::tuple goal, pybind11::list occupancy, pybind11::object cost_map, bool allow_diagonal);
pybind11::tuple astar_to_any_goal(pybind11::tuple start, pybind11::list goal_mask, pybind11::list occupancy, pybind11::object cost_map);
pybind11::list extract_frontiers(pybind11::list occupancy, int unknown_value, int free_value, int occupied_value);
pybind11::list cluster_frontiers(pybind11::list frontier_mask, int min_cluster_size);
int update_coverage_bitset(pybind11::list visited, pybind11::list observed);
double compute_coverage_ratio(pybind11::list visited, pybind11::list valid_free);
pybind11::tuple connected_components(pybind11::list binary_grid);
int raycast_unknown_gain(pybind11::list grid, pybind11::tuple origin, double yaw_rad, int max_range_cells, double fov_rad, int rays, int unknown_value, int occupied_value);
pybind11::dict candidate_feature_summary(pybind11::list grid, pybind11::tuple center, int radius, int unknown_value, int free_value, int occupied_value, int visited_value);
