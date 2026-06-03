#include "grid_planning.hpp"

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
