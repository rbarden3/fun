# %%
from collections import deque

# Given graph
my_graphs = [
    # {
    #     "1": ["6"],
    #     "2": ["3", "6"],
    #     "3": ["2", "4", "5", "6"],
    #     "4": ["3"],
    #     "5": ["3", "6", "7"],
    #     "6": ["1", "2", "3", "5"],
    #     "7": ["5", "8"],
    #     "8": ["7"],
    # },
    # {
    #     "1": ["6"],
    #     "2": ["3", "6"],
    #     "3": ["2", "4", "5", "6"],
    #     "4": ["3"],
    #     "5": ["3", "6", "7"],
    #     "6": ["1", "2", "3", "5"],
    #     "7": ["5", "8"],
    #     "8": ["7", "4"],
    # },
    # {
    #     "1": ["6"],
    #     "2": ["3", "6"],
    #     "3": ["2", "4", "5"],
    #     "4": ["3"],
    #     "5": ["3", "6", "7"],
    #     "6": ["1", "2", "5"],
    #     "7": ["5", "8"],
    #     "8": ["7"],
    # },
    {
        "1": ["6"],
        "2": ["3", "6"],
        "3": ["2", "4", "5"],
        "4": ["9"],
        "5": ["3", "6", "7"],
        "6": ["1", "2", "5"],
        "7": ["5", "8"],
        "8": ["7", "4"],
        "9": ["3"],
    },
]
graphs = [
    # # Simple Undirected Graph (Cyclic)
    # {
    #     'A': ['B', 'C'],
    #     'B': ['A', 'D'],
    #     'C': ['A'],
    #     'D': ['B']
    # },
    # # Directed Graph with Cycles
    # {
    #     'A': ['B'],
    #     'B': ['C'],
    #     'C': ['A'],
    #     'D': ['B'],
    #     'E': []
    # },
    # # Directed Acyclic Graph (DAG)
    # {
    #     'A': ['B', 'C'],
    #     'B': ['D'],
    #     'C': ['D'],
    #     'D': []
    # },
    # # Dense Undirected Graph
    # {
    #     'A': ['B', 'C', 'D', 'E'],
    #     'B': ['A', 'C', 'D', 'E'],
    #     'C': ['A', 'B', 'D', 'E'],
    #     'D': ['A', 'B', 'C', 'E'],
    #     'E': ['A', 'B', 'C', 'D']
    # },
    # # Sparse Directed Graph
    # {
    #     'A': ['B'],
    #     'B': ['C'],
    #     'C': ['D'],
    #     'D': ['E'],
    #     'E': []
    # },
    # # Directed Graph with Multiple Sources and Sinks
    # {
    #     'A': ['B', 'C'],
    #     'B': ['D'],
    #     'C': ['D'],
    #     'D': ['E'],
    #     'E': []
    # },
    # # Bipartite Graph (Undirected)
    # {
    #     'A': ['B', 'C'],
    #     'B': ['A', 'D'],
    #     'C': ['A', 'E'],
    #     'D': ['B'],
    #     'E': ['C']
    # },
    # # Graph with Self-Loops
    # {
    #     'A': ['A', 'B'],
    #     'B': ['A', 'C'],
    #     'C': ['B', 'C'],
    #     'D': ['D'],
    #     'E': ['E']
    # },
    # # Directed Graph with Multiple Edges
    # {
    #     'A': ['B', 'C'],
    #     'B': ['C', 'D'],
    #     'C': ['D'],
    #     'D': []
    # },
    # # Disconnected Graph (Undirected)
    # {
    #     'A': ['B'],
    #     'B': ['A'],
    #     'C': ['D'],
    #     'D': ['C'],
    #     'E': []
    # },
    # # Large Sparse Graph
    # {
    #     'A': ['B'],
    #     'B': ['C'],
    #     'C': ['D'],
    #     'D': ['E'],
    #     'E': ['F'],
    #     'F': ['G'],
    #     'G': ['H'],
    #     'H': ['I'],
    #     'I': ['J'],
    #     'J': []
    # },
    # # Complete Directed Graph
    # {
    #     'A': ['B', 'C', 'D'],
    #     'B': ['A', 'C', 'D'],
    #     'C': ['A', 'B', 'D'],
    #     'D': ['A', 'B', 'C']
    # },
    # # Large Dense Graph (Undirected)
    # {
    #     'A': ['B', 'C', 'D', 'E', 'F'],
    #     'B': ['A', 'C', 'D', 'E', 'F'],
    #     'C': ['A', 'B', 'D', 'E', 'F'],
    #     'D': ['A', 'B', 'C', 'E', 'F'],
    #     'E': ['A', 'B', 'C', 'D', 'F'],
    #     'F': ['A', 'B', 'C', 'D', 'E']
    # },
    # # Directed Graph with Isolated Node
    # {
    #     'A': ['B'],
    #     'B': ['C'],
    #     'C': ['D'],
    #     'D': ['E'],
    #     'E': ['F'],
    #     'F': ['G'],
    #     'G': ['H'],
    #     'H': ['I'],
    #     'I': ['J'],
    #     'J': [],
    #     'K': []  # Isolated node
    # },
    # # Directed Graph with Branches
    # {
    #     'A': ['B', 'C'],
    #     'B': ['D', 'E'],
    #     'C': ['F'],
    #     'D': ['G'],
    #     'E': ['H'],
    #     'F': ['I'],
    #     'G': [],
    #     'H': [],
    #     'I': []
    # },
    # # Directed Graph with a Dead-End Loop
    # {
    #     'A': ['B'],
    #     'B': ['C'],
    #     'C': ['D'],
    #     'D': ['E'],
    #     'E': ['F'],
    #     'F': ['G'],
    #     'G': ['H'],
    #     'H': ['I'],
    #     'I': ['J'],
    #     'J': ['A']  # Creating a dead-end loop from J to A
    # },
    # # Directed Graph with Varying Out-Degree
    # {
    #     'A': ['B', 'C'],
    #     'B': ['D'],
    #     'C': [],
    #     'D': ['E'],
    #     'E': ['F'],
    #     'F': [],
    #     'G': ['A', 'D']
    # }
]

graphs = [*graphs, *my_graphs]


# Helper function to calculate the Euclidean distance
def euclidean_distance(pos1, pos2):
    return (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2


# Helper function to find adjacent empty spaces
def get_adjacent_spaces(occupied: list[tuple[int, int]]) -> set[tuple[int, int]]:
    adj_spaces = set()
    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (1, 1),
        (-1, 1),
        (1, -1),
    ]  # Surrounding Spaces
    for x, y in occupied:
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in occupied:
                adj_spaces.add((nx, ny))
    return adj_spaces


def get_position_score(pos, connected_nodes_pos):
    total_distance = 0
    for connected_node, connected_pos in connected_nodes_pos.items():
        total_distance += euclidean_distance(pos, connected_pos)
    return total_distance


def get_best_position(node_positions, nodes, node):
    # Get adjacent spaces
    adj_spaces = get_adjacent_spaces(node_positions.values())
    connected_nodes_pos = {k: v for k, v in node_positions.items() if k in nodes[node]}

    potential_moves = {
        pos: get_position_score(pos, connected_nodes_pos) for pos in adj_spaces
    }
    best_position = min(potential_moves, key=potential_moves.get)

    return best_position


# DFS-based function to determine node placement order
def dfs(graph, node, visited, node_order, sort):
    node_order.append(node)
    visited.add(node)
    for neighbor in sorted(graph[node], key=lambda x: len(graph[x]), reverse=sort):
        if neighbor not in visited:
            dfs(graph, neighbor, visited, node_order, sort)


def bfs(graph, node, sort):
    visited = []
    queue = deque([node])
    visited.append(node)
    while queue:
        node = queue.popleft()
        # print(node)
        for neighbor in sorted(graph[node], key=lambda x: len(graph[x]), reverse=sort):
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
    return visited


# Algorithm to place nodes on a 2D grid
def place_nodes_on_grid(nodes, dfs_sort=True, sort=True):
    node_positions = {}

    # Start placing the first node in the grid
    start_node = max(nodes, key=lambda x: len(nodes[x]))
    if dfs_sort:
        node_order = []
        visited = set()
        dfs(nodes, start_node, visited, node_order, sort)
    else:
        node_order = list(bfs(nodes, start_node, sort))
    # print(node_order)

    node_positions[start_node] = (0, 0)

    # Place remaining nodes
    for node in node_order[1:]:
        node_positions[node] = get_best_position(node_positions, nodes, node)

    return node_positions


def build_grid(node_positions):
    # Get the min/max x and y values to determine grid size
    x_coords, y_coords = zip(*node_positions.values())
    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))

    # Define the grid size based on the node positions
    grid_width = max_x - min_x + 1
    grid_height = max_y - min_y + 1

    # Initialize the grid with empty spaces (" ")
    grid = [[" " for _ in range(grid_width)] for _ in range(grid_height)]
    # Place nodes in the grid, shifting to accommodate the min_x and min_y offset
    for node, (x, y) in node_positions.items():
        grid[int(y - min_y)][int(x - min_x)] = node

    return grid


def avg_graph_distance(node_positions, nodes):
    distances = {n: 0 for n in nodes}
    for node, neighbors in nodes.items():
        for neighbor in neighbors:
            if neighbor in node_positions:
                distances[node] += euclidean_distance(
                    node_positions[node], node_positions[neighbor]
                )
    return sum(distances.values()) / len(nodes)


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


# Call the function to place nodes
testing_results = {
    "dfs_true": [],
    "dfs_false": [],
    "bfs_true": [],
    "bfs_false": [],
}
for nodes in graphs:
    nodes = fill_nodes(nodes)
    node_positions = {
        "dfs_true": avg_graph_distance(place_nodes_on_grid(nodes), nodes),
        "dfs_false": avg_graph_distance(
            place_nodes_on_grid(nodes, dfs_sort=True, sort=False), nodes
        ),
        "bfs_true": avg_graph_distance(
            place_nodes_on_grid(nodes, dfs_sort=False, sort=True), nodes
        ),
        "bfs_false": avg_graph_distance(
            place_nodes_on_grid(nodes, dfs_sort=False, sort=False), nodes
        ),
    }
    for k, v in node_positions.items():
        testing_results[k].append(v)

    # grid = build_grid(node_positions)
    print(node_positions)

for k, v in testing_results.items():
    print(k, sum(v) / len(v))
# Print the final grid and node positions
# print("Node Positions:", node_positions)
# print("Grid Layout:")
# # print(grid)
for nodes in graphs:
    nodes = fill_nodes(nodes)
    grid = build_grid(place_nodes_on_grid(nodes))
for row in grid:
    for v in row:
        print(v, end=" ")
    print()

# print("Total Graph Distance:", total_graph_distance(node_positions, nodes))
# %%

def convert_format(in_graph):
    """
    Example input format:     
    data = {
        "step": {
            "nodes": [
                {"id": 1},
                {"id": 2},
                {"id": 3},
                {"id": 4},
                {"id": 5},
            ],
            "links": [
                {"source": 1, "target": 2},
                {"source": 1, "target": 3},
                {"source": 4, "target": 5},
            ],
        }
    }

    Example output format:
    {
        "1": ["2", "3"],
        "2": [],
        "3": [],
        "4": ["5"],
        "5": []
    """
    nodes = {str(node["id"]): [] for node in in_graph["step"]["nodes"]}
    for link in in_graph["step"]["links"]:
        nodes[str(link["source"])].append(str(link["target"]))
    return nodes