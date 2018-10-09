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
    return M


def sybil_weights(targets, M, target_dist, Total_stake, walk_len, k=0.0001):
    len_targets = len(targets)

    t = lil_matrix((M.shape[0], len_targets), dtype=np.float64)
    for i, target in enumerate(targets):
        t[target, i] = 1.0
    
    Mp = M.transpose()
    for _ in range(int(walk_len)):
        t = Mp * t

    ret = []

    for i, target in enumerate(targets):
        Y = (t[:,i].A - target_dist) * M.shape[0]
        y_data_init = list(sorted(Y))    
        Logistic = 1.0 / (1.0 + e**(-k*Y)) # Total_stake*
        Y = Logistic
        ret += [(target, Y)]

    from matplotlib import pyplot as plt
    y_data = list(sorted(Y))
    plt.plot(y_data)
    #plt.ylim(0.40, 0.60)
    plt.hlines([0.51, 0.5, 0.49], 0, len(y_data))
    plt.show()

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
                min_quorums[ni] = new_min
                changed = True
                break

    len_good = len(set(node_q) - bad_nodes)
    all_good = True
    for ni in set(node_q) - bad_nodes:
        all_good &= min_quorums[ni] > int(math.floor(len_good/2) + 1)

    return all_good, ref_quorums, min_quorums, just_good_nodes

def stats_cutoff(Scored_nodes, G, cut_off):
    orig_good = set([n for n,_, y in Scored_nodes if y >= cut_off])
    orig_bad = set([n for n,_, y in Scored_nodes if y < cut_off])

    in_good, in_bad, in_between = 0, 0, 0
    for (u, v) in G.edges:
        if u in orig_good and v in orig_good:
            in_good += 1
        elif u in orig_bad and v in orig_bad:
            in_bad += 1
        else:
            in_between += 1 

    return (cut_off, in_good, len(orig_good), in_bad, len(orig_bad), in_between)



def main():
    G = nx.read_edgelist("samples/small_net2.txt", nodetype=int, create_using=nx.DiGraph())
    deg = G.out_degree()

    seed = int(1000000 * random.random())
    print("Random seed: %s" % seed)
    random.seed(seed)

    Good_stake = sum(v for _, v in deg)
    print("Good Stake: %s" % Good_stake)

    none = {
        "num_sybils"  : 1 , 
        "stake_sybils": 0 , 
        "frac_naive"  : 0.5, 
        "stake_naive" : 6 
    }


    light = {
        "num_sybils"  : int(math.ceil(len(G.nodes)/4)) - 1 , 
        "stake_sybils": int(0.50 * Good_stake / 4) - 1, 
        "frac_naive"  : 0.5, 
        "stake_naive" : int(0.50 * Good_stake / 4 ) - 1 
    }

    byzantine = {
        "num_sybils"  : int(math.ceil(len(G.nodes)/2)) - 1 , 
        "stake_sybils": int(0.50 * Good_stake / 2) - 1, 
        "frac_naive"  : 1.0, 
        "stake_naive" : int(0.50 * Good_stake / 2 ) - 1 
    }

    fair = {
        "num_sybils"  : int(math.ceil(len(G.nodes)/2)) - 1 , 
        "stake_sybils": int(0.50 * Good_stake / 2) - 1, 
        "frac_naive"  : 0.1, 
        "stake_naive" : int(0.50 * Good_stake / 2 ) - 1 
    }


    normal = {
        "num_sybils"  : len(G.nodes), 
        "stake_sybils": Good_stake , 
        "frac_naive"  : 0.2, 
        "stake_naive" : int(0.1 * Good_stake) 
    }


    heavy = {
        "num_sybils"  : int(2 * len(G.nodes)) , 
        "stake_sybils": int( 2 * Good_stake) , 
        "frac_naive"  : 0.5, 
        "stake_naive" : int(0.1 * Good_stake) 
    }

    G, node_sets = connect_sybils(G, **normal)
    (Good_nodes, Naive_nodes, Sybil_nodes) = node_sets
    num_naive = len(Naive_nodes)

    Total_stake, Stake_all, _ = state_stats(G, node_sets)
    target_stake, target_dist = compute_targets(Stake_all, Total_stake, node_sets)
    M = compute_matrix(G, node_sets)

    walk_len = 15 * log(M.shape[0]) / log(3)
    print("Walk length: %s" % int(walk_len))
    
    goodlen = len(Good_nodes)
    # logger = []
    Good_set = set(Good_nodes)

    all_nodes = Good_nodes + Sybil_nodes

    Node_samples = 50
    Bad_samples = int(math.ceil(Node_samples * (len(all_nodes) - goodlen) / len(Good_nodes))) - 1
    print("Initial honest: %s" % Node_samples)
    targets = random.sample(list(range(goodlen)), Node_samples)
    target_sybils = random.sample(list(range(goodlen, len(all_nodes))), Bad_samples  )
    
    k = 1
    print("Logistic: k: %d" % k)
    Dists = sybil_weights(targets, M, target_dist, Total_stake, walk_len, k=k)
    
    # Run an analysis to find out who cannot make a good quorum.
    node_q = {}
    bad_nodes = set(target_sybils)
    cut_off = 0.50
    print("Cutoff: %2.2f" % cut_off)

    # Do a connectivity analysis
    target_0, Y_0 = Dists[0]
    Scored_nodes = list(zip(all_nodes, range(len(Y_0)), Y_0))

    for cf in np.arange(0.40, 0.60, 0.01):
        stats = stats_cutoff(Scored_nodes, G, cut_off=cf)
        print("%2.2f: %d (%d), %d (%d) : %d" % stats)

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

if __name__ == "__main__":
    main() 