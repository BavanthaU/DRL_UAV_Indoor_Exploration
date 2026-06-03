#include "grid_planning.hpp"

#include <queue>
#include <set>
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

int int_at(const py::list& grid, int row, int col) {
    return grid[row].cast<py::list>()[col].cast<int>();
}

bool bool_at(const py::list& grid, int row, int col) {
    return grid[row].cast<py::list>()[col].cast<bool>();
}

}  // namespace

py::list extract_frontiers(py::list occupancy, int unknown_value, int free_value, int occupied_value) {
    (void)occupied_value;
    const int rows = n_rows(occupancy);
    const int cols = n_cols(occupancy);
    py::list mask;
    for (int row = 0; row < rows; ++row) {
        py::list out_row;
        for (int col = 0; col < cols; ++col) {
            bool frontier = false;
            if (int_at(occupancy, row, col) == free_value) {
                for (const auto& [dr, dc] : std::vector<std::pair<int, int>>{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}) {
                    const int nr = row + dr;
                    const int nc = col + dc;
                    if (nr >= 0 && nc >= 0 && nr < rows && nc < cols && int_at(occupancy, nr, nc) == unknown_value) {
                        frontier = true;
                        break;
                    }
                }
            }
            out_row.append(frontier);
        }
        mask.append(out_row);
    }
    return mask;
}

py::list cluster_frontiers(py::list frontier_mask, int min_cluster_size) {
    const int rows = n_rows(frontier_mask);
    const int cols = n_cols(frontier_mask);
    std::set<std::pair<int, int>> remaining;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            if (bool_at(frontier_mask, row, col)) {
                remaining.insert({row, col});
            }
        }
    }

    py::list clusters;
    while (!remaining.empty()) {
        const auto start = *remaining.begin();
        remaining.erase(start);
        std::queue<std::pair<int, int>> queue;
        std::vector<std::pair<int, int>> cluster;
        queue.push(start);
        cluster.push_back(start);
        while (!queue.empty()) {
            const auto current = queue.front();
            queue.pop();
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    if (dr == 0 && dc == 0) {
                        continue;
                    }
                    const std::pair<int, int> next{current.first + dr, current.second + dc};
                    const auto found = remaining.find(next);
                    if (found != remaining.end()) {
                        remaining.erase(found);
                        queue.push(next);
                        cluster.push_back(next);
                    }
                }
            }
        }
        if (static_cast<int>(cluster.size()) >= min_cluster_size) {
            py::list out_cluster;
            for (const auto& cell : cluster) {
                out_cluster.append(py::make_tuple(cell.first, cell.second));
            }
            clusters.append(out_cluster);
        }
    }
    return clusters;
}
