# %%
import networkx as nx
import numpy as np

# Input graph
nodes = {
    "1": ["6"],
    "2": ["3", "6"],
    "3": ["2", "4", "5", "6"],
    "4": ["3", "8"],
    "5": ["3", "6", "7"],
    "6": ["1", "2", "3", "5"],
    "7": ["5", "8"],
    "8": ["7","4"],
}


def remove_empty_rows(grid):
    for i in range(2):
        for row in grid:
            if all(not v for v in row):
                row.clear()
        grid = [row for row in grid if row]
        grid = list(map(list, zip(*grid)))
    return grid


# Step 5: Create a grid representation
def generate_grid(grid_positions):
    # Get the min/max x and y values to determine grid size
    x_coords, y_coords = zip(*grid_positions.values())
    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))

    # Define the grid size based on the node positions
    grid_width = max_x - min_x + 1
    grid_height = max_y - min_y + 1

    # Initialize the grid with empty spaces (" ")
    grid = [[" " for _ in range(grid_width)] for _ in range(grid_height)]
    # Place nodes in the grid, shifting to accommodate the min_x and min_y offset
    for node, (x, y) in grid_positions.items():
        grid[int(y - min_y)][int(x - min_x)] = node

    return grid


def get_grid_positions(grid):
    grid_positions = {}
    for y, row in enumerate(grid):
        for x, val in enumerate(row):
            if val != " ":
                grid_positions[val] = (x, y)
    return grid_positions


def euclidean_distance(p1, p2):
    """Compute Euclidean distance between two points."""
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2




# %%
# Step 1: Create a graph
G = nx.Graph()
for node, neighbors in nodes.items():
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

# Step 2: Generate the layout using spring_layout
# initial_positions = nx.spring_layout(G, seed=42)

# initial_positions = nx.arf_layout(G, seed=42, pos=initial_positions)
# initial_positions = nx.spectral_layout(G)
# initial_positions = nx.planar_layout(G)
# initial_positions = nx.circular_layout(G)
initial_positions = nx.nx_pydot.pydot_layout(G, prog="dot")
initial_positions = nx.nx_pydot.graphviz_layout(G)
# initial_positions = nx.kamada_kawai_layout(G, pos=initial_positions)
# initial_positions = nx.forceatlas2_layout(G, pos=initial_positions,distributed_action=True, linlog=False, seed=42, scaling_ratio=100)
# initial_positions = nx.arf_layout(G, seed=42, pos=initial_positions)
# nx.draw(G, pos=initial_positions)
nx.draw_networkx(G, pos=initial_positions)

# %%

# Step 3: Snap each point to the grid
# grid_positions = {
#     node: (round(pos[0] * 10), round(pos[1] * 10))
#     for node, pos in initial_positions.items()
# }
# print(grid_positions)

# # Step 4: Minimize the Euclidean distance between connected nodes
# # grid_positions = minimize_distances(G, grid_positions)
# # print(grid_positions)


# grid = generate_grid(grid_positions)

# # Display the grid with empty spaces as " "
# for row in grid:
#     print(" ".join(row))


# # %%
# # Display the grid with empty spaces as " "
# for row in generate_grid(get_grid_positions(grid)):
#     print(" ".join(row))
