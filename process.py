import networkx as nx


import os
arr = os.listdir("samples")


for fname in arr:
    G = nx.read_edgelist("samples/%s" % fname, nodetype=int, create_using=nx.DiGraph())
    print("Process %s ..." % fname)
    print("Initial # of nodes: %d" % len(G.nodes))

    while True:
        D = G.out_degree()
        nodes_to_remove = [n for n, d  in D if d < 4]
        if len(nodes_to_remove) == 0:
            break
        G.remove_nodes_from(nodes_to_remove)

    print("Final # of nodes: %d" % len(G.nodes))
    nx.write_edgelist(G, "samples/small_%s" % fname)