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

# Use a force-directed layout algorithm to embed the graph in 2D
pos = nx.spring_layout(G)

# Convert positions to a 2D array
node_coordinates = np.array([pos[node] for node in sorted(G.nodes())])

print("2D Array Representation (Node Coordinates):")
print(node_coordinates)

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
grid = [["None" for _ in range(grid_width)] for _ in range(grid_height)]

# Map nodes to grid positions
x_offset = -min(x_coords)
y_offset = -min(y_coords)
for node, (x, y) in normalized_positions.items():
    grid[y + y_offset][x + x_offset] = node
# %%
# Print the 2D array
for row in grid:
    for v in row:
        if v == "None":
            print(" ", end=" ")
        else:
            print(v, end=" ")
    print()

# %%
import networkx as nx
import numpy as np

# Input dictionary
nodes = {'1': ['6'], 
         '2': ['3', '6'],
         '3': ['2', '4', '5', '6'],
         '4': ['3'],
         '5': ['3', '6', '7'],
         '6': ['1', '2', '3', '5'],
         '7': ['5', '8'],
         '8': ['7']}

# Step 1: Create a graph
G = nx.Graph()
for node, neighbors in nodes.items():
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

# Step 2: Compute a layout
positions = nx.spring_layout(G)  # Positions are in normalized continuous space

# Step 3: Map positions to a grid
scale_factor = 10  # Scale factor to make positions integers
grid_positions = {node: (int(pos[0] * scale_factor), int(pos[1] * scale_factor)) for node, pos in positions.items()}

# Step 4: Create a grid
# Determine grid size
x_coords, y_coords = zip(*grid_positions.values())
grid_width = max(x_coords) - min(x_coords) + 1
grid_height = max(y_coords) - min(y_coords) + 1

# Initialize the grid
grid = [[None for _ in range(grid_width)] for _ in range(grid_height)]

# Place nodes in the grid
for node, (x, y) in grid_positions.items():
    grid[y - min(y_coords)][x - min(x_coords)] = node


# Print the 2D array
for row in grid:
    for v in row:
        if not v or v == "None":
            print(" ", end=" ")
        else:
            print(v, end=" ")
    print()

# %%
from collections import deque

# Input dictionary
nodes = {'1': ['6'], 
         '2': ['3', '6'],
         '3': ['2', '4', '5', '6'],
         '4': ['3'],
         '5': ['3', '6', '7'],
         '6': ['1', '2', '3', '5'],
         '7': ['5', '8'],
         '8': ['7']}

# Directions for grid placement: up, down, left, right
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# BFS to place nodes on the grid
def place_nodes(graph):
    grid = {}  # Stores node positions as {(row, col): node}
    visited = set()  # Track visited nodes
    queue = deque([('1', (0, 0))])  # Start with an arbitrary node at (0, 0)
    grid[(0, 0)] = '1'
    visited.add('1')
    
    while queue:
        current_node, position = queue.popleft()
        row, col = position
        
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                # Try placing the neighbor in adjacent cells
                for dr, dc in directions:
                    new_pos = (row + dr, col + dc)
                    if new_pos not in grid:  # Find the first free position
                        grid[new_pos] = neighbor
                        visited.add(neighbor)
                        queue.append((neighbor, new_pos))
                        break
    
    return grid

# Place nodes on the grid
grid_mapping = place_nodes(nodes)

# Determine grid bounds
min_row = min(pos[0] for pos in grid_mapping)
max_row = max(pos[0] for pos in grid_mapping)
min_col = min(pos[1] for pos in grid_mapping)
max_col = max(pos[1] for pos in grid_mapping)

# Create a 2D grid
grid = [[None for _ in range(max_col - min_col + 1)] for _ in range(max_row - min_row + 1)]

# Fill the 2D grid with node positions
for (row, col), node in grid_mapping.items():
    grid[row - min_row][col - min_col] = node

# Display the grid
for row in grid:
    for v in row:
        if not v or v == "None":
            print(" ", end=" ")
        else:
            print(v, end=" ")
    print()

# %%
import networkx as nx
import numpy as np

# Input dictionary
nodes = {'1': ['6'], 
         '2': ['3', '6'],
         '3': ['2', '4', '5', '6'],
         '4': ['3'],
         '5': ['3', '6', '7'],
         '6': ['1', '2', '3', '5'],
         '7': ['5', '8'],
         '8': ['7']}

# Step 1: Create a graph
G = nx.Graph()
for node, neighbors in nodes.items():
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

# Step 2: Compute a force-directed layout
positions = nx.spring_layout(G)  # Positions are in continuous 2D space

# Scale positions to fit a grid
scale_factor = 10
scaled_positions = {node: (int(pos[0] * scale_factor), int(pos[1] * scale_factor)) for node, pos in positions.items()}

# Step 3: Resolve conflicts by shifting nodes to nearby free cells
def resolve_conflicts(scaled_positions):
    occupied = {}
    resolved_positions = {}

    for node, (x, y) in scaled_positions.items():
        if (x, y) not in occupied:
            # Place the node at its intended position
            occupied[(x, y)] = node
            resolved_positions[node] = (x, y)
        else:
            # Find the nearest free position
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    new_pos = (x + dx, y + dy)
                    if new_pos not in occupied:
                        occupied[new_pos] = node
                        resolved_positions[node] = new_pos
                        break
    return resolved_positions

resolved_positions = resolve_conflicts(scaled_positions)

# Step 4: Create a grid representation
x_coords, y_coords = zip(*resolved_positions.values())
min_x, max_x = min(x_coords), max(x_coords)
min_y, max_y = min(y_coords), max(y_coords)

grid_width = max_x - min_x + 1
grid_height = max_y - min_y + 1

# Initialize the grid
grid = [[None for _ in range(grid_width)] for _ in range(grid_height)]

# Place nodes in the grid
for node, (x, y) in resolved_positions.items():
    grid[y - min_y][x - min_x] = node

# Display the grid
for row in grid:
    for v in row:
        if not v or v == "None":
            print(" ", end=" ")
        else:
            print(v, end=" ")
    print()

# %%
import networkx as nx
import numpy as np

# Input graph
nodes = {'1': ['6'], 
         '2': ['3', '6'],
         '3': ['2', '4', '5', '6'],
         '4': ['3'],
         '5': ['3', '6', '7'],
         '6': ['1', '2', '3', '5'],
         '7': ['5', '8'],
         '8': ['7']}

# Directions for adjacency (including diagonals)
directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# Step 1: Create a graph
G = nx.Graph()
for node, neighbors in nodes.items():
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

# Step 2: Compute force-directed layout for Euclidean positioning
positions = nx.spring_layout(G, iterations=10000)  # Continuous positions
scale_factor = 10
scaled_positions = {node: (int(pos[0] * scale_factor), int(pos[1] * scale_factor)) for node, pos in positions.items()}
print(scaled_positions)
# Step 3: Place nodes and adjust for compactness
def place_nodes_compactly(graph, scaled_positions):
    grid = {}
    occupied = set()
    node_connections = {node: len(neighbors) for node, neighbors in graph.items()}

    for node, (x, y) in scaled_positions.items():
        if (x, y) not in occupied:
            grid[(x, y)] = node
            occupied.add((x, y))
        else:
            # Resolve conflicts by placing nearby
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    new_pos = (x + dx, y + dy)
                    if new_pos not in occupied:
                        grid[new_pos] = node
                        occupied.add(new_pos)
                        break

    # Adjust compactness
    for (x, y), node in list(grid.items()):
        connected_nodes = node_connections[node]
        adjacent = [(x + dx, y + dy) for dx, dy in directions]
        empty_spaces = sum(1 for pos in adjacent if pos not in grid)
        
        if empty_spaces > 8 - connected_nodes:
            # Relocate node to a denser area
            for dx, dy in directions:
                new_pos = (x + dx, y + dy)
                if new_pos not in grid:
                    del grid[(x, y)]
                    grid[new_pos] = node
                    break

    return grid

# Apply placement algorithm
resolved_positions = place_nodes_compactly(nodes, scaled_positions)

# Step 4: Create a grid
x_coords, y_coords = zip(*resolved_positions.keys())
min_x, max_x = min(x_coords), max(x_coords)
min_y, max_y = min(y_coords), max(y_coords)

grid_width = max_x - min_x + 1
grid_height = max_y - min_y + 1

grid = [[None for _ in range(grid_width)] for _ in range(grid_height)]

for (x, y), node in resolved_positions.items():
    grid[y - min_y][x - min_x] = node

# Step 5: Print the grid
for row in grid:
    for v in row:
        if not v or v == "None":
            print(" ", end=" ")
        else:
            print(v, end=" ")
    print()

# %%
import networkx as nx

# Input graph
nodes = {'1': ['6'], 
         '2': ['3', '6'],
         '3': ['2', '4', '5', '6'],
         '4': ['3'],
         '5': ['3', '6', '7'],
         '6': ['1', '2', '3', '5'],
         '7': ['5', '8'],
         '8': ['7']}

def custom_layout(graph):
    """
    Compute a layout where nodes are placed on a grid prioritizing left/right/up/down.
    """
    # Step 1: Compute initial positions using spring_layout
    initial_positions = nx.spring_layout(graph)
    
    # Step 2: Snap positions to a grid
    grid_positions = {}
    for node, (x, y) in initial_positions.items():
        # Snap to nearest integer grid while preferring left/right or up/down
        snapped_x = round(x)
        snapped_y = round(y)
        grid_positions[node] = (snapped_x, snapped_y)
    
    # Step 3: Resolve conflicts by preferring horizontal/vertical adjustments
    occupied = {}
    resolved_positions = {}
    for node, (x, y) in grid_positions.items():
        if (x, y) not in occupied:
            # Place node if space is available
            resolved_positions[node] = (x, y)
            occupied[(x, y)] = node
        else:
            # Resolve conflicts by shifting horizontally or vertically
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_pos = (x + dx, y + dy)
                if new_pos not in occupied:
                    resolved_positions[node] = new_pos
                    occupied[new_pos] = node
                    break

    return resolved_positions

# Step 1: Create a graph
G = nx.Graph()
for node, neighbors in nodes.items():
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

# Step 2: Apply the custom layout
resolved_positions = custom_layout(G)

# Step 3: Create a grid representation
x_coords, y_coords = zip(*resolved_positions.values())
min_x, max_x = min(x_coords), max(x_coords)
min_y, max_y = min(y_coords), max(y_coords)

grid_width = max_x - min_x + 1
grid_height = max_y - min_y + 1

grid = [[None for _ in range(grid_width)] for _ in range(grid_height)]

for node, (x, y) in resolved_positions.items():
    grid[y - min_y][x - min_x] = node

# Display the grid
for row in grid:
    for v in row:
        if not v or v == "None":
            print(" ", end=" ")
        else:
            print(v, end=" ")
    print()

# %%
import networkx as nx
import numpy as np

# Input graph
nodes = {'1': ['6'], 
         '2': ['3', '6'],
         '3': ['2', '4', '5', '6'],
         '4': ['3'],
         '5': ['3', '6', '7'],
         '6': ['1', '2', '3', '5'],
         '7': ['5', '8'],
         '8': ['7']}

def euclidean_distance(p1, p2):
    """Compute Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def custom_layout(graph):
    """
    Compute a layout where nodes are placed on a grid while minimizing Euclidean distance.
    We will use a force-directed layout (spring_layout) and snap to a grid with left/right and up/down preference.
    """
    # Step 1: Compute initial positions using spring_layout
    initial_positions = nx.spring_layout(graph, seed=42)  # Set a random seed for reproducibility

    # Step 2: Store the original distances between connected nodes
    original_distances = {}
    for node in graph.nodes():
        for neighbor in graph.neighbors(node):
            dist = euclidean_distance(initial_positions[node], initial_positions[neighbor])
            original_distances[(node, neighbor)] = dist
            original_distances[(neighbor, node)] = dist
    
    # Step 3: Snap positions to a grid
    grid_positions = {}
    for node, (x, y) in initial_positions.items():
        # Snap to nearest integer grid
        snapped_x = round(x)
        snapped_y = round(y)
        grid_positions[node] = (snapped_x, snapped_y)
    
    # Step 4: Resolve conflicts by preferring horizontal/vertical adjustments
    occupied = {}
    resolved_positions = {}
    for node, (x, y) in grid_positions.items():
        if (x, y) not in occupied:
            # Place node if space is available
            resolved_positions[node] = (x, y)
            occupied[(x, y)] = node
        else:
            # Resolve conflicts by shifting horizontally or vertically
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_pos = (x + dx, y + dy)
                if new_pos not in occupied:
                    resolved_positions[node] = new_pos
                    occupied[new_pos] = node
                    break

    # Step 5: Check distances again, and adjust positions to minimize the Euclidean distance between connected nodes
    for (node, neighbor), original_dist in original_distances.items():
        if node in resolved_positions and neighbor in resolved_positions:
            current_dist = euclidean_distance(resolved_positions[node], resolved_positions[neighbor])
            if current_dist > original_dist:
                # If the current distance is greater than the original distance, adjust positions
                node_x, node_y = resolved_positions[node]
                neighbor_x, neighbor_y = resolved_positions[neighbor]
                # Adjust positions to bring them closer
                new_node_pos = ((node_x + neighbor_x) / 2, (node_y + neighbor_y) / 2)
                new_neighbor_pos = new_node_pos
                resolved_positions[node] = new_node_pos
                resolved_positions[neighbor] = new_neighbor_pos

    return resolved_positions

# Step 1: Create a graph
G = nx.Graph()
for node, neighbors in nodes.items():
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

# Step 2: Apply the custom layout
resolved_positions = custom_layout(G)

# Step 3: Create a grid representation
# Get the min/max x and y values to determine grid size
x_coords, y_coords = zip(*resolved_positions.values())
min_x, max_x = min(x_coords), max(x_coords)
min_y, max_y = min(y_coords), max(y_coords)

# Ensure grid dimensions are integers
grid_width = int(max_x - min_x + 1)
grid_height = int(max_y - min_y + 1)

# Initialize the grid with empty spaces (" ")
grid = [[" " for _ in range(grid_width)] for _ in range(grid_height)]

# Place nodes in the grid, shifting to accommodate the min_x and min_y offset
for node, (x, y) in resolved_positions.items():
    grid[y - min_y][x - min_x] = node

# Display the grid with empty spaces as " "
for row in grid:
    print(" ".join(row))



# %%
