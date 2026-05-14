import networkx as nx
import numpy as np


def connect_graph(v_num, edges):
    """Make the graph connected by adding a hub vertex.

    The hub is connected to one random vertex in each connected component.
    """
    G = nx.Graph()
    G.add_nodes_from(np.arange(v_num))
    if len(edges) > 0:
        G.add_edges_from(edges)
    components = [
        list(c)
        for c in sorted(nx.connected_components(G), key=len, reverse=True)
    ]
    auxiliary_edges = []
    for component in components:
        auxiliary_edges.append([v_num, component[0]])
    if len(auxiliary_edges) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.stack(auxiliary_edges).astype(np.int32)


def minimum_spanning_tree(v_num, edges, weights):
    """Build minimum spanning tree weighted by uncertainty."""
    G = nx.Graph()
    G.add_nodes_from(np.arange(v_num))
    for i in range(edges.shape[0]):
        G.add_edge(int(edges[i, 0]), int(edges[i, 1]), weight=float(weights[i]))
    T = nx.minimum_spanning_tree(G)
    return list(nx.dfs_preorder_nodes(T, source=0)), nx.dfs_predecessors(T, source=0)
