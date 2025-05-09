# %%
import networkx as nx
import scipy.sparse

# %%
my_nodes = list(range(5))
H = nx.DiGraph()


# %%
H.add_nodes_from(my_nodes)

H.add_edges_from(
    [
        (1,0),
        (1,2),
        (1,3),
        (3,2),
        (3,4),
        (4,0)
    ]
)
# %%
nx.draw(H, with_labels=True)
# %% get dense adj matrix
nx.adjacency_matrix(H).todense()
#%%