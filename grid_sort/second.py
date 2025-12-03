# %%
import networkx as nx
import numpy as np

# Define the graph
graph_dict = {
    '1': ['6'],
    '2': ['3', '6'],
    '3': ['2', '4', '5', '6'],
    '4': ['3'],
    '5': ['3', '6', '7'],
    '6': ['1', '2', '3', '5'],
    '7': ['5', '8'],
    '8': ['7']
}

# Create a NetworkX graph
G = nx.Graph()
for node, neighbors in graph_dict.items():
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

# Use a force-directed layout to compute positions
pos = nx.spring_layout(G)

# Normalize positions to a grid
# Scale coordinates to fit in a 2D integer grid
grid_scale = 10  # Scaling factor for grid size
normalized_positions = {
    node: (int(grid_scale * x), int(grid_scale * y)) 
    for node, (x, y) in pos.items()
}

# Determine the grid size
x_coords, y_coords = zip(*normalized_positions.values())
grid_width = max(x_coords) - min(x_coords) + 1
grid_height = max(y_coords) - min(y_coords) + 1

# Create a blank grid filled with "None"
grid = [[None for _ in range(grid_width)] for _ in range(grid_height)]

# Map nodes to grid positions
x_offset = -min(x_coords)
y_offset = -min(y_coords)
for node, (x, y) in normalized_positions.items():
    grid[y + y_offset][x + x_offset] = node

for i in range(2):
    for row in grid:
        if all(not v for v in row):
            row.clear()
    grid = [row for row in grid if row]
    grid = list(map(list, zip(*grid)))
    

# %%
# Print the 2D array
for row in grid:
    for v in row:
        if not v:
            print(" ", end=" ")
        else:
            print(v, end=" ")
    print()
# %%
