import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sla

G = nx.gnp_random_graph(100, 0.1)
nx.draw(G)

# wait for the user input
input("Press Enter to continue...")