# %%
from copy import deepcopy
from itertools import chain

import numpy as np

# %%
nodes = {
    "1": set(["6"]),
    "2": set(["3", "6"]),
    "3": set(["4", "5", "6"]),
    "4": set([]),
    "5": set(["3", "6", "7"]),
    "6": set([]),
    "7": set(["8"]),
}
# I think we can get a good layout by walking down the tree, and then walking back up
# Walking down the tree, we use the parent graphs as guides to place the subnodes
# Walking back up the tree, we use the subgraphs to place the parent nodes, and remove the subgraph ranges from being editable
subgraphs = {
    "4": {"4.1": set(["4.2"]), "4.2": set(["3"])},
    "5": {"5.1": set(["3", "6", "7"]), "5.2": set([]), "5.3": set(["3"])},
}


def fill_nodes(nodes):
    out = {**nodes}
    for node, neighbors in nodes.items():
        for neighbor in neighbors:
            if neighbor not in out:
                out[neighbor] = set()
            out[neighbor].add(node)

    return {k: sorted(v) for k, v in sorted(out.items())}


nodes = fill_nodes(nodes)


# %%
def pad_grid(in_grid, area=2):
    grid = deepcopy(in_grid)
    grid = [
        [None for _ in range(area)] + row + [None for _ in range(area)] for row in grid
    ]
    vert_adds = [[None for _ in range(len(grid[0]))] for _ in range(area)]
    grid = vert_adds + grid + vert_adds
    return grid


# %%
def build_grid(nodes):
    grid = [[None for _ in range(len(nodes))] for _ in range(len(nodes))]
    set_point = len(nodes) // 2
    for ind, node in enumerate(nodes):
        grid[ind][set_point] = node
    return pad_grid(grid)


def print_cell(cell):
    val = str(cell)
    pads = 8 - len(val)
    return f"{val}{' ' * pads}"


def print_grid(grid):
    delim = "-" * len(grid[0]) * 8
    print(delim)
    for row in grid:
        for cell in row:
            print(print_cell(cell), end="")
        print()
    print(delim)


# %%
print_grid(build_grid(nodes))


# %%
def distance(node1, node2):
    slope = (node1[1] - node2[1]) / (node1[0] - node2[0])
    dist = np.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)
    sign = -1 if np.signbit(slope) else 1
    return sign * dist


# %%
def get_distances(nodes, grid):
    distances = {node: [] for node in nodes}
    for node, neighbors in nodes.items():
        node_loc = np.asarray(np.where(np.array(grid) == node)).T[0]
        for neighbor in neighbors:
            neighbor_loc = np.asarray(np.where(np.array(grid) == neighbor)).T[0]
            distances[node].append(float(distance(node_loc, neighbor_loc)))
    return distances


# %%
grid = build_grid(nodes)


# %%


def get_surrounding_cells(node, grid):
    area = 1
    node_loc = np.asarray(np.where(np.array(grid) == node)).T[0]
    row_above = max(node_loc[0] - area, 0)
    row_below = min(node_loc[0] + area, len(grid) - 1)
    col_left = max(node_loc[1] - area, 0)
    col_right = min(node_loc[1] + area, len(grid[0]) - 1)

    for r in range(row_above, row_below + 1):
        for c in range(col_left, col_right + 1):
            if r == node_loc[0] and c == node_loc[1]:
                continue
            yield (r, c)


def get_boundry(grid):
    out = [
        ((0, c) for c in range(len(grid[0]))),
        ((len(grid) - 1, c) for c in range(len(grid[0]))),
        ((r, 0) for r in range(len(grid))),
        ((r, len(grid[0]) - 1) for r in range(len(grid))),
    ]
    return set(chain.from_iterable(out))


# %%
def get_best_shift(node, in_grid, nodes, look_ahead=1):
    grid = deepcopy(in_grid)
    distances = get_distances(nodes, grid)
    best = {"move": None, "dist": distances[node], "grid": in_grid}

    possible_moves = (
        (r, c) for r, c in get_surrounding_cells(node, grid) if grid[r][c] == None
    )

    for move in possible_moves:
        move_grid = shift_node(node, grid, move)
        move_dist = get_distances(nodes, move_grid)[node]
        test = {
            "look_ahead": look_ahead,
            "node": node,
            "move": move,
            "dist": move_dist,
        }
        if abs_sum(move_dist) < abs_sum(best["dist"]):
            best["move"] = move
            best["dist"] = move_dist
            best["grid"] = move_grid
        elif look_ahead > 0:
            for connection in nodes[node]:
                ahead = find_move(connection, move_grid, nodes, look_ahead - 1)
                move_grid = ahead["grid"]
                move_dist = get_distances(nodes, move_grid)[node]
                if abs_sum(move_dist) < abs_sum(best["dist"]):
                    best["move"] = move
                    best["dist"] = move_dist
                    best["grid"] = move_grid

    return best


def shift_node(node, in_grid, position):
    grid = deepcopy(in_grid)
    num_rows = len(grid)
    node_loc = np.asarray(np.where(np.array(grid) == node)).T[0]
    grid[node_loc[0]][node_loc[1]] = None
    grid[position[0]][position[1]] = node
    grid = [row for row in grid if any(row)]
    while len(grid) < num_rows:
        grid.append([None for _ in range(len(grid[0]))])
    return grid


# %%
def compare_distances(dist, others):
    for other in others:
        if abs_sum(other) < abs_sum(dist):
            return False
    return True


def find_move(node, in_grid, nodes, look_ahead=1):
    grid = deepcopy(in_grid)
    distances = get_distances(nodes, grid)
    node_loc = np.asarray(np.where(np.array(grid) == node)).T[0]
    node_loc = (int(node_loc[0]), int(node_loc[1]))

    best_shift = get_best_shift(node, grid, nodes, look_ahead)
    best_swap = get_best_swap(node, grid, nodes, look_ahead)
    moves = {}
    moves["shift"] = abs_sum(best_shift["dist"])
    moves["swap"] = abs_sum(best_swap["dist"])
    out = {"node": node, "move_type": None, "from": node_loc, "to": None, "grid": grid}

    if min(moves.values()) >= abs_sum(distances[node]):
        if look_ahead > 0:
            print(f"{node}: no move")
        return out

    if all(moves["shift"] <= v for k, v in moves.items() if k != "shift"):
        out["to"] = best_shift["move"]
        out["move_type"] = "shift"
        out["grid"] = best_shift["grid"]
    elif all(moves["swap"] <= v for k, v in moves.items() if k != "swap"):
        out["to"] = best_swap["move"]
        out["move_type"] = "swap"
        out["grid"] = best_swap["grid"]

    return out


def swap_nodes(node_pos, neighbor_pos, in_grid):
    grid = deepcopy(in_grid)
    node_val = grid[node_pos[0]][node_pos[1]]
    neighbor_val = grid[neighbor_pos[0]][neighbor_pos[1]]
    grid[node_pos[0]][node_pos[1]] = neighbor_val
    grid[neighbor_pos[0]][neighbor_pos[1]] = node_val
    return grid


def get_best_swap(node, in_grid, nodes, look_ahead=1):
    grid = deepcopy(in_grid)
    distances = get_distances(nodes, grid)
    best = {"move": None, "dist": distances[node], "grid": deepcopy(in_grid)}
    node_loc = np.asarray(np.where(np.array(grid) == node)).T[0]

    possible_moves = (
        (r, c) for r, c in get_surrounding_cells(node, grid) if grid[r][c] != None
    )

    for move in possible_moves:
        move_grid = swap_nodes(node_loc, move, grid)
        move_dist = get_distances(nodes, move_grid)[node]
        if abs_sum(move_dist) < abs_sum(best["dist"]):
            best["move"] = move
            best["dist"] = move_dist
            best["grid"] = move_grid
    return best


# %%


def abs_sum(iterable):
    return sum((abs(val) for val in iterable))


def abs_avg(iterable):
    return abs_sum(iterable) / len(iterable)


# %%
def trim_grid(grid):
    out = [row[:] for row in grid if any(row)]
    out = [*zip(*out)]
    out = [row for row in out if any(row)]
    out = [*zip(*out)]
    return out


# %%
