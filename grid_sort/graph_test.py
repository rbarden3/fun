# %%
import networkx as nx
import numpy as np

# Input graph
nodes = {
    "1": ["6"],
    "2": ["3", "6"],
    "3": ["2", "4", "5", "6"],
    "4": ["3"],
    "5": ["3", "6", "7"],
    "6": ["1", "2", "3", "5"],
    "7": ["5", "8"],
    "8": ["7"],
}


# Step 1: Create a graph
G = nx.Graph()
for node, neighbors in nodes.items():
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

initial_positions = nx.spectral_layout(G)

initial_positions = nx.kamada_kawai_layout(G, pos=initial_positions)


nx.draw_networkx(G, pos=initial_positions)


# %%
