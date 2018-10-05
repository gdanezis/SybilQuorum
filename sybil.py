import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
import random
from math import log, e
import math


def connect_sybils(G, num_sybils, stake_sybils, frac_naive, stake_naive, verbose=True):
    if verbose:
        print("  --- Graph ---")
        print("Number Good: %d" % len(G.nodes))
        print("Number Sybils: %d" % num_sybils)
        print("Stake Sybils:  %d" % stake_sybils)
        print("Fraction Naive:  %2.3f" % frac_naive)
        print("Stake Naive:  %d" % stake_naive)
        print("  -------------")
    

    G = G.copy()
    Good_nodes = list(G.nodes)
    random.shuffle(Good_nodes)
    Sybil_nodes = list(range(10000000, 10000000 + num_sybils))

    #assert stake_sybils % 2 == 0 and stake_naive % 2 == 0

    # Attach Sybil nodes
    # First links within Sybils

    Bad_stake = stake_sybils
    while Bad_stake > 1:
        [s0, s1] = random.sample(Sybil_nodes, 2)
        if not G.has_edge(s0, s1):
            G.add_edge(s0, s1)
            G.add_edge(s1, s0)
            Bad_stake -= 2

    # Then add links to the "Good" region
    fraction_of_naive_nodes = frac_naive
    num_naive = int(fraction_of_naive_nodes * len(Good_nodes))
    Naive_nodes = Good_nodes[:num_naive]
    # print("Naive nodes %s out of %s" % (len(Naive_nodes), len(Good_nodes)))
    Bad_stake = stake_naive
    while Bad_stake > 1:
        s0 = random.choice(Sybil_nodes)
        n0 = random.choice(Naive_nodes)
        if not G.has_edge(s0, n0):
            G.add_edge(s0, n0)
            G.add_edge(n0, s0)
            Bad_stake -= 2


    Sybil_nodes = [s for s in Sybil_nodes if s in G.nodes]
    return G, (Good_nodes, Naive_nodes, Sybil_nodes)


# Compute the stake per node:
def state_stats(G, node_sets):
    (Good_nodes, Naive_nodes, Sybil_nodes) = map(set, node_sets)
    Node_stake = G.out_degree()
    Within_good = 0
    Within_Sybil = 0 
    Good_to_bad = 0
    Bad_to_Good = 0

    for u,v in G.edges():
        if u in Good_nodes and v in Good_nodes:
            Within_good += 1
        elif u not in Good_nodes and v not in Good_nodes:
            Within_Sybil += 1
        elif u in Good_nodes and v not in Good_nodes:
            Good_to_bad += 1
        else:
            Bad_to_Good += 1

    Total_stake = Within_good + Within_Sybil + Good_to_bad + Bad_to_Good
    print("Stake: Good: %s to_Bad: %s Bad: %s to_Good: %s" % (Within_good, Good_to_bad, Within_Sybil, Bad_to_Good))
    return Total_stake, Node_stake, (Within_good, Within_Sybil, Good_to_bad, Bad_to_Good)


def compute_targets(Stake_all, Total_stake, node_sets):
    (Good_nodes, Naive_nodes, Sybil_nodes) = node_sets
    target_stake = np.zeros((len(Good_nodes + Sybil_nodes), 1))
    for s, n in enumerate(Good_nodes + Sybil_nodes):
        try:
            target_stake[s,0] = float(Stake_all[n]) 
        except:
            target_stake[s,0] = 0.0

    target_dist = target_stake / Total_stake
    # print(np.sum(target_dist))
    assert round(np.sum(target_dist), 3) == 1.0
    return target_stake, target_dist


def compute_matrix(G, node_sets):
    (Good_nodes, Naive_nodes, Sybil_nodes) = node_sets
    deg = G.out_degree()
    V_all = len(G.edges())
    for n0 in G.nodes:
        d0 = deg[n0]
        W = 0.0
        for n1 in G[n0]:

            w = 1.0 / d0
            G[n0][n1]["weight"] = w
            W += w
        assert round(W, 2) == 1.0

    M = nx.adjacency_matrix(G, Good_nodes + Sybil_nodes)
    # print("Shape:", M.shape)
    return M


def sybil_weights(targets, M, target_dist, Total_stake, walk_len, k=10):
    len_targets = len(targets)

    t = lil_matrix((M.shape[0], len_targets), dtype=np.float64)
    for i, target in enumerate(targets):
        t[target, i] = 1.0
    
    Mp = M.transpose()
    for _ in range(int(walk_len)):
        t = Mp * t

    ret = []

    for i, target in enumerate(targets):
        Y = (t[:,i].A - target_dist)
        Logistic = 1.0 / (1.0 + e**(-k*Total_stake*Y))
        Y = Logistic
        ret += [(target, Y)]
    return ret

def test_all_quorums_for_size(node_q, bad_nodes):
    # Premise: all nodes that are not bad have quorums that intersect.
    # They do so if they contain more than good_nodes - the min_quorum size.

    just_good_nodes = set(node_q) - bad_nodes

    min_quorums = {}
    for ni in just_good_nodes:

        min_quorums[ni] = int(math.floor(2 * (len(node_q[ni])) / 3) + 1 - len(node_q[ni] & bad_nodes))

    ref_quorums = min_quorums.copy()       
    changed = True
    while changed:
        changed = False
        for ni in just_good_nodes:
            new_min = list(sorted(min_quorums[nj] for nj in node_q[ni] - bad_nodes))[ref_quorums[ni]]

            # Since a node needs to contain other nodes in its quorum it must
            if new_min > min_quorums[ni]:
                #print(ni, min_quorums[ni], "->", new_min)
                min_quorums[ni] = new_min
                changed = True
                break

    len_good = len(set(node_q) - bad_nodes)
    all_good = True
    for ni in set(node_q) - bad_nodes:
        all_good &= min_quorums[ni] > int(math.floor(len_good/2) + 1)

    return all_good, ref_quorums, min_quorums, just_good_nodes


def main():
    G = nx.read_edgelist("samples/small_net2.txt", nodetype=int, create_using=nx.DiGraph())
    deg = G.out_degree()

    seed = int(1000000 * random.random())
    print("Random seed: %s" % seed)
    random.seed(seed)

    Good_stake = sum(v for _, v in deg)
    print("Good Stake: %s" % Good_stake)

    light = {
        "num_sybils"  : int(math.ceil(len(G.nodes)/4)) - 1 , 
        "stake_sybils": int(0.50 * Good_stake / 4) - 1, 
        "frac_naive"  : 0.5, 
        "stake_naive" : int(0.50 * Good_stake / 4 ) - 1 
    }

    fair = {
        "num_sybils"  : int(math.ceil(len(G.nodes)/2)) - 1 , 
        "stake_sybils": int(0.50 * Good_stake / 2) - 1, 
        "frac_naive"  : 1.0, 
        "stake_naive" : int(0.50 * Good_stake / 2 ) - 1 
    }

    heavy = {
        "num_sybils"  : int(2 * len(G.nodes)) , 
        "stake_sybils": int( 2 * Good_stake) , 
        "frac_naive"  : 0.5, 
        "stake_naive" : int(0.1 * Good_stake) 
    }

    #num_sybils, stake_sybils, frac_naive, stake_naive):
    G, node_sets = connect_sybils(G, **heavy)
    (Good_nodes, Naive_nodes, Sybil_nodes) = node_sets
    num_naive = len(Naive_nodes)

    Total_stake, Stake_all, _ = state_stats(G, node_sets)
    target_stake, target_dist = compute_targets(Stake_all, Total_stake, node_sets)
    M = compute_matrix(G, node_sets)

    walk_len = 15 * log(M.shape[0]) / log(3)
    print("Walk length: %s" % int(walk_len))
    
    goodlen = len(Good_nodes)
    logger = []
    Good_set = set(Good_nodes)

    # A Confused node connects more to sybils than honest nodes.
    #confused = set()
    #for ni in Good_nodes:
    #    sc_good = 0
    #    sc_bad  = 0 
    #    for nj in G[ni]:
    #        if nj in Good_set:
    #            sc_good += 1
    #        else:
    #            sc_bad += 1

    #    if not (sc_good >= 2 * sc_bad + 1):
    #        confused.add(ni)


    all_nodes = Good_nodes + Sybil_nodes

    Node_samples = 50
    Bad_samples = int(math.ceil(Node_samples * (len(all_nodes) - goodlen) / len(Good_nodes))) - 1
    print("Initial honest: %s" % Node_samples)
    targets = random.sample(list(range(goodlen)), Node_samples)
    target_sybils = random.sample(list(range(goodlen, len(all_nodes))), Bad_samples  )
    
    k = 10
    print("Logistic: k: %d" % k)
    Dists = sybil_weights(targets, M, target_dist, Total_stake, walk_len, k=k)

    # print("\nNode Health:")
    #healthy, total = 0, 0 

    #actual_targets = [target for target, Y in Dists]
    
    # Run an analysis to find out who cannot make a good quorum.
    node_q = {}
    bad_nodes = set(target_sybils)
    cut_off = 0.49
    print("Cutoff: %2.2f" % cut_off)

    for target, Y in Dists:
        node_q[target] = set()
        for i in targets + target_sybils:
            if Y[i] >= cut_off:
                node_q[target].add( i )

        n_good = len([n for n in node_q[target] if n not in bad_nodes])
        n_all = len(node_q[target])

        print(target, n_all, "%2.2f" % (100*n_good/n_all))
    

    new_bad_nodes = bad_nodes | set()
    while len(new_bad_nodes) > 0:
        old_bad_nodes = new_bad_nodes
        new_bad_nodes = set()

        for ni in set(node_q) - bad_nodes:
            if not len(node_q[ni] - bad_nodes) > 2 * len(node_q[ni] & bad_nodes):
                bad_nodes.add(ni)
                new_bad_nodes.add(ni)
                # print(ni, len(node_q[ni] - bad_nodes), 2 * len(node_q[ni] & bad_nodes))

    # print("Total bad nodes: %d out of: %d nodes" % (len(bad_nodes), len(targets + target_sybils)))
    all_good, _, _, just_good_nodes = test_all_quorums_for_size(node_q, bad_nodes)
    print(all_good, len(just_good_nodes))

    print("  --- Results ---")
    print("Pure Good: %d" % len(just_good_nodes))
    print(" Confused: %d" % (Node_samples - len(just_good_nodes)))
    print("   Sybils: %d" % Bad_samples)
    print("Agreement: %s" % all_good)
    print("  ---------------")


    return


    for target, Y in sorted(Dists):

        # Implement the threshold approach to selecting nodes.
        cut_off = 0.50
        selec_good = set()
        selec_bad  = set()
        
        #node_index = dict((node, idx) for idx, node in enumerate(Good_nodes + Sybil_nodes))

        for i in targets + target_sybils:
            if Y[i] >= cut_off:
                if i not in bad_nodes:
                    selec_good.add( all_nodes[i] )
                else:
                    selec_bad.add( all_nodes[i] )

        pc_good = 100.0 * float(len(selec_good)) / (len(selec_good) + len(selec_bad))

        # Log the number of nodes that are actually good, 
        # ie. surrounded by at most 1/3 bad nodes (in stake)

        conf = ["", "*"][all_nodes[target] in confused]
        no_quorum = ["", "*"][pc_good < 67]
        
        total += 1
        if pc_good > 66:
            healthy += 1

        logger += [(target, selec_good, selec_bad, all_nodes[target] in confused)]

        # print("--")
        print("%d%s\t%3d%%%s\t%d\t%d" % (target, conf, pc_good, no_quorum, len(selec_good) + len(selec_bad), len(targets + target_sybils)))
    
    print("Overall: %2.2f%% healthy" % (100.0 * float(healthy) / total))
    print("Confused: %2.2f%%" % (100.0 * len(confused) / len(Good_nodes)))
    print()
    print("Quorum Intersection health:")
    # Test last for quorum intersection

    # (2) Quorum Availability despite B

    V = {n for n in Good_nodes if n not in confused}
    for (other, other_good, other_bad, other_comp) in logger:
        conf = [" ", "c"][all_nodes[other] in confused]
        # print ( len(other_good) > 2 * len(other_bad), conf) 

    # (1) Quorum Intersection despite B

    G = nx.Graph()

    links  =  10 * len(logger)
    while links > 0:
        log1, log2 = random.sample(logger, 2)
        last, last_good, last_bad, last_comp = log1
        other, other_good, other_bad, other_comp = log2

        if last in bad_nodes or other in bad_nodes:
            continue    

        links -= 1

        union_size = len(last_good | other_good)
        z0 = 2 * (len(other_good) + len(other_bad)) / 3 - len(other_bad)
        z1 = 2 * (len(last_good) + len(last_bad)) / 3 - len(last_bad)
        # print(z0 > number_just_good / 2 + 1, z1 > number_just_good / 2 + 1,)
        flag1 = ["!", "-"][len(other_good) > 2 * len(other_bad)]
        flag2 = ["!", "-"][len(last_good) > 2 * len(other_bad)]

        sum_z = round(z0 + z1)
        print("%d\t%d\t%s\t%d\t>?\t%d\t%s" % (other, last, sum_z > union_size, sum_z, union_size, flag1+flag2))
        if sum_z > union_size:
            G.add_edge(other, last)

    comp = [len(ns) for ns in nx.connected_components(G)] 
    num_good = len(set(targets + target_sybils) - bad_nodes)

    print(comp, num_good, len(comp) >0 and comp[0] == num_good)
    print("Missing:", [n for n in targets if n not in bad_nodes and n not in G.nodes])
    try:
        print(nx.diameter(G))
    except:
        print("No diameter")



    #    is_quorum = 3* len(other_good) > 2*len(other_good | other_bad) + 1 
    #    qlen = round(2.0 / 3.0 + max( len(other_good) + len(other_bad),  len(last_good) + len(last_bad)) / 3)
    #    print(qlen, " < ", len(last_good & other_good), qlen < len(last_good & other_good), is_quorum, len(other_good | other_bad),   last_comp, other_comp, conf)

    if False:
        naive = Y[:num_naive]
        honest  = Y[num_naive:goodlen]
        Sybils = Y[goodlen:]

        bins = np.arange(-0.05, 1.15, 0.1)

        from matplotlib import pyplot as plt
        fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
        axs[0].hist(honest, bins=bins, label="Honest")
        axs[0].title.set_text('Honest')
        axs[1].hist(naive, bins=bins, label="Naive")
        axs[1].title.set_text('Naive')
        axs[2].hist(Sybils, bins=bins, label="Sybils")
        axs[2].title.set_text('Sybil')
        plt.show()

        # print(X)

if __name__ == "__main__":
    main() 