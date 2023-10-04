import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Add nodes
G.add_nodes_from([
    ("Generate Customer Comment", {"shape": "box"}),
    ("Generate Email Subject", {"shape": "box"}),
    ("Generate Summary of Customer Comment", {"shape": "box"}),
    ("Translate Comment and Email Summary", {"shape": "box"}),
    ("Analyze Sentiment of Customer's Comment", {"shape": "box"}),
    ("Generate Email to Customer", {"shape": "box"}),
])

# Add edges
G.add_edges_from([
    ("Generate Customer Comment", "Generate Email Subject"),
    ("Generate Customer Comment", "Generate Summary of Customer Comment"),
    ("Generate Summary of Customer Comment", "Translate Comment and Email Summary"),
    ("Translate Comment and Email Summary", "Analyze Sentiment of Customer's Comment"),
    ("Analyze Sentiment of Customer's Comment", "Generate Email to Customer"),
])

# Positioning of nodes
pos = {
    "Generate Customer Comment": (1, 6),
    "Generate Email Subject": (1, 4),
    "Generate Summary of Customer Comment": (1, 2),
    "Translate Comment and Email Summary": (4, 2),
    "Analyze Sentiment of Customer's Comment": (7, 2),
    "Generate Email to Customer": (10, 2),
}

# Draw the graph
nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=2000, node_color='skyblue', font_size=8)
plt.show()
