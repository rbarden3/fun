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
ext_nodes_dict = {"5": {"3": (-1, 0), "6": (0, 1), "7": (1, 0)}, "4": {"3": (1, 0)}}


def fill_nodes(nodes, add_missing=True):
    out = {**nodes}
    for node, neighbors in nodes.items():
        for neighbor in neighbors:
            if neighbor not in out and add_missing:
                out[neighbor] = set()

            if neighbor in out and type(out[neighbor]) != set:
                out[neighbor] = set(out[neighbor])
            if neighbor in out:
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
    grid = [[None for _ in range(len(nodes))] for _ in range(len(nodes) + 2)]
    set_point = len(nodes) // 2
    for ind, node in enumerate(nodes):
        grid[ind + 1][set_point] = node
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
    diff_a = node1[0] - node2[0]
    diff_b = node1[1] - node2[1]
    sign_a = 1 if diff_a >= 0 else -1
    sign_b = 1 if diff_b >= 0 else -1

    sign = sign_a * sign_b

    dist = math.sqrt((diff_a**2) + (diff_b**2))
    return sign * dist


# %%
def get_distances(nodes, grid, ext_nodes=None):
    distances = {node: [] for node in nodes}
    for node, neighbors in nodes.items():
        node_loc = find_node(grid, node)
        for neighbor in neighbors:
            if ext_nodes and neighbor in ext_nodes:
                x, y = ext_nodes[neighbor]
                if x == 0:
                    x = len(grid) // 2
                elif x == 1:
                    x = len(grid)
                if y == 0:
                    y = len(grid[0]) // 2
                elif y == 1:
                    y = len(grid[0])
                neighbor_loc = (x, y)

                distances[node].append(float(distance(node_loc, neighbor_loc)))
                continue

            neighbor_loc = find_node(grid, neighbor)
            distances[node].append(float(distance(node_loc, neighbor_loc)))
    return distances


# %%
grid = build_grid(nodes)


# %%
def center_row(row):
    vals = [cell for cell in row if cell]
    empties = len(row) - len(vals)
    pad = empties // 2
    end = empties - pad
    return [None for _ in range(pad)] + vals + [None for _ in range(end)]


def find_node(grid, node):
    for i, row in enumerate(grid):
        if node in row:
            return (i, row.index(node))


def center_nodes(grid):
    vals = [center_row(row) for row in grid if any(row)]
    empties = len(grid) - len(vals)
    pad = empties // 2
    end = empties - pad
    return add_row(grid, pad) + vals + add_row(grid, end)


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


def get_shift_moves(connections, grid, buffer=1):
    out = set()
    for node in connections:
        for move in get_surrounding_cells(node, grid, area=buffer):
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
def get_best_shift(node, in_grid, nodes, look_ahead=1, ext_nodes=None):
    grid = deepcopy(in_grid)
    distances = get_distances(nodes, grid, ext_nodes)
    best = {"move": None, "dist": distances[node], "grid": in_grid}

    # possible_moves = get_empty_cells(grid)
    # possible_moves = (cell for cell in get_surrounding_cells(node, grid) if grid[cell[0]][cell[1]] == None)
    buffer = 1
    if ext_nodes:
        graph_nodes = [node for node in nodes[node] if node not in ext_nodes]
        buffer = 2
    else:
        graph_nodes = nodes[node]
    possible_moves = get_shift_moves([*graph_nodes, node], grid, buffer)

    for move in possible_moves:
        move_grid = shift_node(node, grid, move)
        move_dist = get_distances(nodes, move_grid, ext_nodes)[node]
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
            for connection in graph_nodes:
                ahead = find_move(
                    connection, move_grid, nodes, look_ahead - 1, ext_nodes
                )
                move_grid = ahead["grid"]
                move_dist = get_distances(nodes, move_grid, ext_nodes)[node]
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


def find_move(node, in_grid, nodes, look_ahead=1, ext_nodes=None):
    grid = deepcopy(in_grid)
    # grid = pad_grid(trim_grid(grid))
    distances = get_distances(nodes, grid, ext_nodes)
    node_loc = find_node(grid, node)

    best_shift = get_best_shift(node, grid, nodes, look_ahead, ext_nodes)
    best_swap = get_best_swap(node, grid, nodes, look_ahead, ext_nodes)
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

    # if any_edge_nodes(out["grid"]):
    #     out["grid"] = center_nodes(out["grid"])
    return out


def swap_nodes(node_pos, neighbor_pos, in_grid):
    grid = deepcopy(in_grid)
    node_val = grid[node_pos[0]][node_pos[1]]
    neighbor_val = grid[neighbor_pos[0]][neighbor_pos[1]]
    grid[node_pos[0]][node_pos[1]] = neighbor_val
    grid[neighbor_pos[0]][neighbor_pos[1]] = node_val
    return grid


def get_best_swap(node, in_grid, nodes, look_ahead=1, ext_nodes=None):
    grid = deepcopy(in_grid)
    distances = get_distances(nodes, grid, ext_nodes)
    best = {"move": None, "dist": distances[node], "grid": deepcopy(in_grid)}
    node_loc = find_node(grid, node)

    possible_moves = get_node_cells(grid).values()

    for move in possible_moves:
        move_grid = swap_nodes(node_loc, move, grid)
        move_dist = get_distances(nodes, move_grid, ext_nodes)[node]
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
    grid, nodes, ext_nodes=None, look_ahead=1, max_iterations=100, max_depth=2
):
    sorted_grid = [row[:] for row in grid]
    moves = {}
    last_moves = []
    go = True
    count = 0
    single_connections = (node for node in nodes if len(nodes[node]) == 1)
    for node in single_connections:
        moves[node] = find_move(node, sorted_grid, nodes, look_ahead, ext_nodes)
        sorted_grid = moves[node]["grid"]
        if not moves[node]["move_type"]:
            print(f"{node}: no move")
        else:
            print({k: v for k, v in moves[node].items() if k != "grid"})
            print_grid(sorted_grid)
    while go:
        for node in nodes:
            moves[node] = find_move(node, sorted_grid, nodes, look_ahead, ext_nodes)
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
    return sorted_grid


# %%
# %%
# nodes = {
#     "1": set(["6"]),
#     "2": set(["3", "6"]),
#     "3": set(["4", "5", "6"]),
#     "4": set([]),
#     "5": set(["3", "6", "7"]),
#     "6": set([]),
#     "7": set(["8"]),
# }
# # I think we can get a good layout by walking down the tree, and then walking back up
# # Walking down the tree, we use the parent graphs as guides to place the subnodes
# # Walking back up the tree, we use the subgraphs to place the parent nodes, and remove the subgraph ranges from being editable
# subgraphs = {
#     "4": {"4.1": set(["4.2"]), "4.2": set(["3"])},
#     "5": {"5.1": set(["3", "6", "7"]), "5.2": set([]), "5.3": set(["3"])},
# }
# ext_nodes = {"5": {"3": (-1, 0), "6": (0, 1), "7": (1, 0)}, "4": {"3": (1, 0)}}


# %%
def get_rel_pos(origin_node, other_node, grid):
    origin_loc = find_node(grid, origin_node)
    other_loc = find_node(grid, other_node)
    r = other_loc[0] - origin_loc[0]
    c = other_loc[1] - origin_loc[1]

    r = 0 if r == 0 else r // abs(r)
    c = 0 if c == 0 else c // abs(c)
    return (r, c)


# %%
def count_empty_slices(grid):
    counts = {"t": 0, "b": 0, "l": 0, "r": 0}
    for row in grid:
        if all(cell == None for cell in row):
            counts["t"] += 1
        else:
            break
    for row in reversed(grid):
        if all(cell == None for cell in row):
            counts["b"] += 1
        else:
            break
    for col in range(len(grid[0])):
        if all(row[col] == None for row in grid):
            counts["l"] += 1
        else:
            break
    for col in reversed(range(len(grid[0]))):
        if all(row[col] == None for row in grid):
            counts["r"] += 1
        else:
            break
    return counts


# %%
def row_add_count(grid, trimmed, r_mid, c_mid, r_start, c_start):

    row_finder = list(reversed(range(r_mid)))
    add_top = len(row_finder)
    for r in row_finder:
        if any(grid[r_start + r][c] for c in range(c_start, c_start + len(trimmed[0]))):
            break
        add_top -= 1

    row_finder = list(range(r_mid + 1, len(trimmed)))
    add_bottom = len(row_finder)
    for r in row_finder:
        if any(grid[r_start + r][c] for c in range(c_start, c_start + len(trimmed[0]))):
            break
        add_bottom -= 1

    row_finder = list(reversed(range(c_mid)))
    add_left = len(row_finder)
    for c in row_finder:
        if any(
            grid[r][c_start + c]
            for r in range(r_start + add_top, r_start + len(trimmed) - add_bottom)
        ):
            break
        add_left -= 1

    row_finder = list(range(c_mid + 1, len(trimmed[0])))
    add_right = len(row_finder)
    for c in row_finder:
        if any(
            grid[r][c_start + c]
            for r in range(r_start + add_top, r_start + len(trimmed) - add_bottom)
        ):
            break
        add_right -= 1

    return {"t": add_top, "b": add_bottom, "l": add_left, "r": add_right}


#  %%
def insert_rows(in_grid, add_rows, node_loc):
    grid = deepcopy(in_grid)
    grid = (
        grid[: node_loc[0] + 1] + add_row(grid, add_rows["b"]) + grid[node_loc[0] + 1 :]
    )
    grid = grid[: node_loc[0]] + add_row(grid, add_rows["t"]) + grid[node_loc[0] :]
    grid = [
        row[: node_loc[1] + 1] + [None] * add_rows["r"] + row[node_loc[1] + 1 :]
        for row in grid
    ]
    grid = [
        row[: node_loc[1]] + [None] * add_rows["l"] + row[node_loc[1] :] for row in grid
    ]
    return grid


# %%
def insert_subgrid(in_grid, in_subgrid, node):
    grid = deepcopy(in_grid)
    subgrid = deepcopy(in_subgrid)
    trimmed = trim_grid(subgrid)
    node_loc = find_node(grid, node)
    grid[node_loc[0]][node_loc[1]] = None

    empty_slices = count_empty_slices(subgrid)
    r_mid = len(trimmed) // 2
    c_mid = len(trimmed[0]) // 2
    if len(trimmed) % 2 == 0:
        if empty_slices["t"] > empty_slices["b"]:
            r_mid -= 1

    if len(trimmed[0]) % 2 == 0:
        if empty_slices["l"] > empty_slices["r"]:
            c_mid -= 1

    r_start = node_loc[0] - r_mid
    c_start = node_loc[1] - c_mid

    add_rows = row_add_count(grid, trimmed, r_mid, c_mid, r_start, c_start)
    grid = insert_rows(grid, add_rows, node_loc)
    node_loc = (node_loc[0] + add_rows["t"], node_loc[1] + add_rows["l"])
    r_start = node_loc[0] - r_mid
    c_start = node_loc[1] - c_mid
    tl_corner = (r_start, c_start)
    br_corner = (r_start + len(trimmed) - 1, c_start + len(trimmed[0]) - 1)

    for r, row in enumerate(trimmed):
        for c, cell in enumerate(row):
            grid[r_start + r][c_start + c] = cell
    return {"grid": grid, "tl": tl_corner, "br": br_corner}


# %%
def run_alg(nodes, subgraphs):
    nodes = fill_nodes(nodes)
    grid = build_grid(nodes)
    grid = optimize_grid(grid, nodes, None)
    # grid = trim_grid(grid)
    grid_nodes = set(chain.from_iterable(grid))
    subgrids = {}
    subgrid_ranges = {}

    made_changes = True

    while made_changes:
        inner_changes = False
        search_subgraphs = {
            k: v for k, v in subgraphs.items() if k not in subgrids and k in grid_nodes
        }
        for node, subgraph in search_subgraphs.items():
            print(f"------------------- Subgraph {node}-------------------")
            ext_node_list = [
                ext_node
                for ext_node in chain.from_iterable(subgraph.values())
                if ext_node in grid_nodes
            ]
            ext_nodes = {
                ext_node: get_rel_pos(node, ext_node, grid)
                for ext_node in ext_node_list
            }
            subnodes = fill_nodes(subgraphs[node], False)
            subgrid = build_grid(subnodes)
            subgrids[node] = optimize_grid(subgrid, subnodes, ext_nodes)
            inner_changes = True

        for k in search_subgraphs:
            subgrid_data = insert_subgrid(grid, subgrids[k], k)
            grid = subgrid_data["grid"]
            subgrid_ranges[k] = {"tl": subgrid_data["tl"], "br": subgrid_data["br"]}

        grid_nodes = set(chain.from_iterable(grid))

        made_changes = inner_changes

    trimmed = trim_grid(grid)
    print_grid(trimmed)
    return trimmed


# %%
def get_subgrid_ranges(grid, subgrid):
    grid_nodes = set(chain.from_iterable(grid))
    subgrid_points = []

    for k in subgrid:
        if k in grid_nodes:
            subgrid_points.append(find_node(grid, k))
    subgrid_points = sorted(subgrid_points)
    return {"tl": subgrid_points[0], "br": subgrid_points[-1]}


# %%
def generate_layout(grid, subgrids):
    node_width = 200
    node_height = 100
    padding_x = 50
    padding_y = 50
    parent_padding = 40
    start_x = parent_padding
    start_y = parent_padding
    subgrid_ranges = {k: get_subgrid_ranges(grid, v) for k, v in subgrids.items()}

    layout = []

    subgrid_x_points = set()
    subgrid_y_points = set()

    for subgrid in subgrid_ranges.values():
        subgrid_x_points.add(subgrid["tl"][1])
        subgrid_x_points.add(subgrid["br"][1] + 0.5)
        subgrid_y_points.add(subgrid["tl"][0])
        subgrid_y_points.add(subgrid["br"][0] + 0.5)

    subgrid_x_points = sorted(subgrid_x_points)
    subgrid_y_points = sorted(subgrid_y_points)
    for k, v in subgrid_ranges.items():
        tl = v["tl"]
        br = v["br"]
        previous_x_subgrids = len([x for x in subgrid_x_points if x < tl[1]])
        previous_y_subgrids = len([y for y in subgrid_y_points if y < tl[0]])
        br_x = len([x for x in subgrid_x_points if x <= br[1]]) - 1
        br_y = len([y for y in subgrid_y_points if y <= br[0]]) - 1

        x_diff = br_x - previous_x_subgrids
        y_diff = br_y - previous_y_subgrids

        x = start_x
        x += (tl[1] * (node_width + padding_x)) - parent_padding
        x += previous_x_subgrids * parent_padding

        y = start_y
        y += (tl[0] * (node_height + padding_y)) - parent_padding
        y += previous_y_subgrids * parent_padding
        width = (((br[1] - tl[1] + 1) * node_width) + ((br[1] - tl[1]) * padding_x)) + (
            2 * parent_padding
        )
        width += x_diff * parent_padding
        height = (
            ((br[0] - tl[0] + 1) * node_height) + ((br[0] - tl[0]) * padding_y)
        ) + (2 * parent_padding)
        height += y_diff * parent_padding
        # print(
        #     f"subgrid: {k}, y_diff: {y_diff}, x_diff: {x_diff}, br_x: {br_x}, previous_x_subgrids: {previous_x_subgrids}, br_y: {br_y}, previous_y_subgrids: {previous_y_subgrids}"
        # )
        layout.append(
            {
                "id": f"subgrid_{k}",
                "position": {"x": x, "y": y},
                "style": {
                    "width": width,
                    "height": height,
                },
                "type": "group",
            }
        )
    for r, row in enumerate(grid):
        previous_y_subgrids = len([y for y in subgrid_y_points if y <= r])
        y = 0
        y += r * (node_height + padding_y)
        y += previous_y_subgrids * parent_padding
        for c, cell in enumerate(row):
            if cell:
                previous_x_subgrids = len([x for x in subgrid_x_points if x <= c])
                x = 0
                x += c * (node_width + padding_x)
                x += previous_x_subgrids * parent_padding
                # if cell in ["3", "7", "5.3", "5.2"]:
                #     print(
                #         f"cell: {cell}, r: {r}, c: {c}, previous_x_subgrids: {previous_x_subgrids}, x: {x}, previous_y_subgrids: {previous_y_subgrids}, y: {y}"
                #     )
                #     print(subgrid_y_points)
                layout.append(
                    {
                        "id": cell,
                        "data": {"label": cell},
                        "position": {"x": x, "y": y},
                        "style": {
                            "width": node_width,
                            "height": node_height,
                        },
                    }
                )
    return layout


# %%
