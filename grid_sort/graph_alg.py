# %%
import math
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
def add_row(grid, area):
    return [[None for _ in range(len(grid[0]))] for _ in range(area)]


def pad_grid(in_grid, area=1):
    grid = deepcopy(in_grid)
    grid = [
        [None for _ in range(area)] + row + [None for _ in range(area)] for row in grid
    ]
    grid = add_row(grid, area) + grid + add_row(grid, area)
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
    if node1[0] - node2[0] >= 0:
        sign_a = 1
    else:
        sign_a = -1
    if node1[1] - node2[1] >= 0:
        sign_b = 1
    else:
        sign_b = -1

    sign = sign_a * sign_b

    dist = math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)
    return sign * dist


# %%
def get_distances(nodes, grid):
    distances = {node: [] for node in nodes}
    for node, neighbors in nodes.items():
        node_loc = find_node(grid, node)
        for neighbor in neighbors:
            neighbor_loc = find_node(grid, neighbor)
            distances[node].append(float(distance(node_loc, neighbor_loc)))
    return distances


# %%
grid = build_grid(nodes)


# %%
def center_row(row):
    if any(row):
        vals = [cell for cell in row if cell]
        pad = len(row) // 2
        end = len(row) - len(vals) - pad
        return [None for _ in range(pad)] + vals + [None for _ in range(end)]
    return row


def find_node(grid, node):
    for i, row in enumerate(grid):
        if node in row:
            return (i, row.index(node))


def center_nodes(grid):
    return [center_row(row) for row in grid]


def any_edge_nodes(grid):
    top = grid[0]
    bottom = grid[-1]
    left = (row[0] for row in grid)
    right = (row[-1] for row in grid)
    return any(chain(top, bottom, left, right))


# %%
def get_empty_cells(grid):
    return (
        (r, c)
        for r in range(len(grid))
        for c in range(len(grid[0]))
        if grid[r][c] == None
    )


def get_shift_moves(connections, grid):
    out = set()
    for node in connections:
        for move in get_surrounding_cells(node, grid):
            if grid[move[0]][move[1]] == None:
                out.add(move)
    return out


def get_node_cells(grid):
    cells = (
        (r, c)
        for r in range(len(grid))
        for c in range(len(grid[0]))
        if grid[r][c] != None
    )
    return {grid[cell[0]][cell[1]]: cell for cell in cells}


def get_surrounding_cells(node, grid, area=1):
    node_loc = find_node(grid, node)
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

    # possible_moves = get_empty_cells(grid)
    # possible_moves = (cell for cell in get_surrounding_cells(node, grid) if grid[cell[0]][cell[1]] == None)
    possible_moves = get_shift_moves(nodes[node], grid)

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


# %%
def shift_node(node, in_grid, position):
    grid = deepcopy(in_grid)
    node_loc = find_node(grid, node)
    grid[node_loc[0]][node_loc[1]] = None
    grid[position[0]][position[1]] = node
    return grid


# %%
def compare_distances(dist, others):
    for other in others:
        if abs_sum(other) < abs_sum(dist):
            return False
    return True


def find_move(node, in_grid, nodes, look_ahead=1):
    grid = deepcopy(in_grid)
    # grid = pad_grid(trim_grid(grid))
    distances = get_distances(nodes, grid)
    node_loc = find_node(grid, node)

    best_shift = get_best_shift(node, grid, nodes, look_ahead)
    best_swap = get_best_swap(node, grid, nodes, look_ahead)
    moves = {}
    moves["shift"] = abs_sum(best_shift["dist"])
    moves["swap"] = abs_sum(best_swap["dist"])
    out = {"node": node, "move_type": None, "from": node_loc, "to": None, "grid": grid}

    if min(moves.values()) >= abs_sum(distances[node]):
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
    node_loc = find_node(grid, node)

    possible_moves = get_node_cells(grid).values()

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
def trim_grid(in_grid):
    grid = deepcopy(in_grid)
    out = (row[:] for row in grid if any(row))
    out = (row for row in zip(*out) if any(row))
    return [list(row) for row in zip(*out)]


# %%
def optimize_grid(
    grid, nodes, look_ahead=1, max_iterations=100, max_depth=2, universal=False
):
    sorted_grid = [row[:] for row in grid]
    moves = {}
    last_moves = []
    go = True
    count = 0
    single_connections = (node for node in nodes if len(nodes[node]) == 1)
    for node in single_connections:
        moves[node] = find_move(node, sorted_grid, nodes, look_ahead)
        sorted_grid = moves[node]["grid"]
        if not moves[node]["move_type"]:
            print(f"{node}: no move")
        else:
            print({k: v for k, v in moves[node].items() if k != "grid"})
            print_grid(sorted_grid)
    while go:
        for node in nodes:
            moves[node] = find_move(node, sorted_grid, nodes, look_ahead)
            sorted_grid = moves[node]["grid"]
            if not moves[node]["move_type"]:
                print(f"{node}: no move")
            else:
                print({k: v for k, v in moves[node].items() if k != "grid"})
                print_grid(sorted_grid)

        passer = {k: v["move_type"] for k, v in moves.items()}
        if all(v == "shift" for v in passer.values()):
            print("No more swaps")
            go = False
        elif all(v == "swap" for v in passer.values()):
            print("No more shifts")
            go = False
        elif all(v == None for v in passer.values()):
            print("No moves made")
            go = False
        elif last_moves == list(passer.values()):
            print("No New moves made")
            print("Trying Universal")
            # return optimize_grid(sorted_grid, nodes, look_ahead, 2, 2, True)
            go = False
            count += 1
            look_ahead += 1
            print(f"Look ahead increased to {look_ahead}")
            if look_ahead > max_depth:
                print("Max depth reached")
                go = False
        else:
            count += 1
            last_moves = list(passer.values())
        if count > max_iterations:
            print("Max iterations reached")
            go = False

    trimmed = trim_grid(sorted_grid)
    print_grid(trimmed)
    return trimmed


# %%
