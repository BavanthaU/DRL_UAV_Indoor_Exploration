#include "grid_planning.hpp"

#include <queue>
#include <utility>
#include <vector>

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

py::tuple connected_components(py::list binary_grid) {
    const int rows = n_rows(binary_grid);
    const int cols = n_cols(binary_grid);
    std::vector<std::vector<int>> labels(rows, std::vector<int>(cols, 0));
    int label = 0;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            if (!bool_at(binary_grid, row, col) || labels[row][col] != 0) {
                continue;
            }
            ++label;
            std::queue<std::pair<int, int>> queue;
            queue.push({row, col});
            labels[row][col] = label;
            while (!queue.empty()) {
                const auto current = queue.front();
                queue.pop();
                for (const auto& [dr, dc] : std::vector<std::pair<int, int>>{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}) {
                    const int nr = current.first + dr;
                    const int nc = current.second + dc;
                    if (nr < 0 || nc < 0 || nr >= rows || nc >= cols) {
                        continue;
                    }
                    if (bool_at(binary_grid, nr, nc) && labels[nr][nc] == 0) {
                        labels[nr][nc] = label;
                        queue.push({nr, nc});
                    }
                }
            }
        }
    }

    py::list out;
    for (const auto& row : labels) {
        py::list out_row;
        for (const auto value : row) {
            out_row.append(value);
        }
        out.append(out_row);
    }
    return py::make_tuple(out, label);
}
