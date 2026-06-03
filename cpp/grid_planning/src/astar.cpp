#include "grid_planning.hpp"

#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct Cell {
    int row = 0;
    int col = 0;
};

struct CellHash {
    std::size_t operator()(const Cell& cell) const {
        return (static_cast<std::size_t>(cell.row) << 32) ^ static_cast<std::size_t>(cell.col);
    }
};

bool operator==(const Cell& lhs, const Cell& rhs) {
    return lhs.row == rhs.row && lhs.col == rhs.col;
}

struct QueueItem {
    double priority = 0.0;
    Cell cell;
};

struct QueueItemGreater {
    bool operator()(const QueueItem& lhs, const QueueItem& rhs) const {
        return lhs.priority > rhs.priority;
    }
};

Cell tuple_to_cell(const py::tuple& value) {
    if (value.size() != 2) {
        throw std::runtime_error("Grid cell must contain exactly two values.");
    }
    return Cell{value[0].cast<int>(), value[1].cast<int>()};
}

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

double cost_at(const py::object& cost_map, int row, int col) {
    if (cost_map.is_none()) {
        return 1.0;
    }
    return cost_map.cast<py::list>()[row].cast<py::list>()[col].cast<double>();
}

double heuristic(const Cell& lhs, const Cell& rhs) {
    return std::abs(lhs.row - rhs.row) + std::abs(lhs.col - rhs.col);
}

py::list build_path(
    Cell current,
    const std::unordered_map<Cell, Cell, CellHash>& came_from
) {
    std::vector<Cell> reversed;
    reversed.push_back(current);
    while (came_from.find(current) != came_from.end()) {
        current = came_from.at(current);
        reversed.push_back(current);
    }
    py::list path;
    for (auto it = reversed.rbegin(); it != reversed.rend(); ++it) {
        path.append(py::make_tuple(it->row, it->col));
    }
    return path;
}

py::tuple astar_impl(
    const Cell& start,
    const Cell& goal,
    const py::list& occupancy,
    const py::object& cost_map,
    bool allow_diagonal
) {
    const int rows = n_rows(occupancy);
    const int cols = n_cols(occupancy);
    if (rows == 0 || cols == 0) {
        return py::make_tuple(py::list(), std::numeric_limits<double>::infinity());
    }

    std::vector<std::pair<int, int>> offsets{{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    if (allow_diagonal) {
        offsets.insert(offsets.end(), {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}});
    }

    std::priority_queue<QueueItem, std::vector<QueueItem>, QueueItemGreater> frontier;
    std::unordered_map<Cell, Cell, CellHash> came_from;
    std::unordered_map<Cell, double, CellHash> cost_so_far;

    frontier.push(QueueItem{0.0, start});
    cost_so_far[start] = 0.0;

    while (!frontier.empty()) {
        const Cell current = frontier.top().cell;
        frontier.pop();
        if (current == goal) {
            return py::make_tuple(build_path(current, came_from), cost_so_far.at(current));
        }

        for (const auto& [dr, dc] : offsets) {
            const int nr = current.row + dr;
            const int nc = current.col + dc;
            if (nr < 0 || nc < 0 || nr >= rows || nc >= cols) {
                continue;
            }
            if (int_at(occupancy, nr, nc) == 2) {
                continue;
            }
            const double step = ((dr != 0 && dc != 0) ? std::sqrt(2.0) : 1.0) * cost_at(cost_map, nr, nc);
            const Cell next{nr, nc};
            const double new_cost = cost_so_far.at(current) + step;
            const auto existing = cost_so_far.find(next);
            if (existing == cost_so_far.end() || new_cost < existing->second) {
                cost_so_far[next] = new_cost;
                came_from[next] = current;
                frontier.push(QueueItem{new_cost + heuristic(next, goal), next});
            }
        }
    }

    return py::make_tuple(py::list(), std::numeric_limits<double>::infinity());
}

}  // namespace

py::tuple astar_grid(py::tuple start, py::tuple goal, py::list occupancy, py::object cost_map, bool allow_diagonal) {
    return astar_impl(tuple_to_cell(start), tuple_to_cell(goal), occupancy, cost_map, allow_diagonal);
}

py::tuple astar_to_any_goal(py::tuple start, py::list goal_mask, py::list occupancy, py::object cost_map) {
    const Cell start_cell = tuple_to_cell(start);
    py::list best_path;
    py::object best_goal = py::none();
    double best_cost = std::numeric_limits<double>::infinity();

    for (int row = 0; row < n_rows(goal_mask); ++row) {
        for (int col = 0; col < n_cols(goal_mask); ++col) {
            if (!bool_at(goal_mask, row, col)) {
                continue;
            }
            const auto result = astar_impl(start_cell, Cell{row, col}, occupancy, cost_map, false);
            const double cost = result[1].cast<double>();
            if (cost < best_cost) {
                best_path = result[0].cast<py::list>();
                best_goal = py::make_tuple(row, col);
                best_cost = cost;
            }
        }
    }

    return py::make_tuple(best_path, best_goal, best_cost);
}
