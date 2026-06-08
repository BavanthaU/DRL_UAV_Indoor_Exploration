#include "grid_planning.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

namespace py = pybind11;

namespace {

int n_rows(const py::list& grid) {
    return static_cast<int>(grid.size());
}

int n_cols(const py::list& grid) {
    if (grid.size() == 0) {
        return 0;
    }
    return static_cast<int>(grid[0].cast<py::list>().size());
}

bool bool_at(const py::list& grid, int row, int col) {
    return grid[row].cast<py::list>()[col].cast<bool>();
}

int int_at(const py::list& grid, int row, int col) {
    return grid[row].cast<py::list>()[col].cast<int>();
}

}  // namespace

int update_coverage_bitset(py::list visited, py::list observed) {
    int count = 0;
    for (int row = 0; row < n_rows(visited); ++row) {
        py::list visited_row = visited[row].cast<py::list>();
        for (int col = 0; col < n_cols(visited); ++col) {
            if (bool_at(observed, row, col) && !bool_at(visited, row, col)) {
                visited_row[col] = py::bool_(true);
                ++count;
            }
        }
    }
    return count;
}

double compute_coverage_ratio(py::list visited, py::list valid_free) {
    int valid = 0;
    int covered = 0;
    for (int row = 0; row < n_rows(valid_free); ++row) {
        for (int col = 0; col < n_cols(valid_free); ++col) {
            if (bool_at(valid_free, row, col)) {
                ++valid;
                if (bool_at(visited, row, col)) {
                    ++covered;
                }
            }
        }
    }
    return valid > 0 ? static_cast<double>(covered) / static_cast<double>(valid) : 0.0;
}

int raycast_unknown_gain(py::list grid, py::tuple origin, double yaw_rad, int max_range_cells, double fov_rad, int rays, int unknown_value, int occupied_value) {
    const int rows = n_rows(grid);
    const int cols = n_cols(grid);
    const int origin_row = origin[0].cast<int>();
    const int origin_col = origin[1].cast<int>();
    std::set<std::pair<int, int>> seen_unknown;
    const int ray_count = std::max(1, rays);
    for (int idx = 0; idx < ray_count; ++idx) {
        const double alpha = ray_count == 1 ? 0.5 : static_cast<double>(idx) / static_cast<double>(ray_count - 1);
        const double angle = yaw_rad - 0.5 * fov_rad + alpha * fov_rad;
        const double step_row = std::sin(angle);
        const double step_col = std::cos(angle);
        for (int step = 1; step <= max_range_cells; ++step) {
            const int row = static_cast<int>(std::llround(static_cast<double>(origin_row) + step_row * static_cast<double>(step)));
            const int col = static_cast<int>(std::llround(static_cast<double>(origin_col) + step_col * static_cast<double>(step)));
            if (row < 0 || row >= rows || col < 0 || col >= cols) {
                break;
            }
            const int value = int_at(grid, row, col);
            if (value == occupied_value) {
                break;
            }
            if (value == unknown_value) {
                seen_unknown.insert({row, col});
            }
        }
    }
    return static_cast<int>(seen_unknown.size());
}

py::dict candidate_feature_summary(py::list grid, py::tuple center, int radius, int unknown_value, int free_value, int occupied_value, int visited_value) {
    const int rows = n_rows(grid);
    const int cols = n_cols(grid);
    const int center_row = center[0].cast<int>();
    const int center_col = center[1].cast<int>();
    int unknown = 0;
    int free = 0;
    int occupied = 0;
    int visited = 0;
    for (int row = std::max(0, center_row - radius); row < std::min(rows, center_row + radius + 1); ++row) {
        for (int col = std::max(0, center_col - radius); col < std::min(cols, center_col + radius + 1); ++col) {
            const int value = int_at(grid, row, col);
            unknown += value == unknown_value;
            free += value == free_value;
            occupied += value == occupied_value;
            visited += value == visited_value;
        }
    }
    py::dict counts;
    counts["unknown"] = unknown;
    counts["free"] = free;
    counts["occupied"] = occupied;
    counts["visited"] = visited;
    return counts;
}
