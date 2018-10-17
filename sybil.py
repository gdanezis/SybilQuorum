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
        Y = list(Logistic)
        ret += [(target, Y)]

    #from matplotlib import pyplot as plt
    #y_data = list(sorted(Y))
    #plt.plot(y_data)
    #plt.hlines([0.51, 0.5, 0.49], 0, len(y_data))
    #plt.show()

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
            order = list(sorted(min_quorums[nj] for nj in node_q[ni] - bad_nodes))
            pos = min(len(order) - 1, ref_quorums[ni] - 1)
            if pos >= 0:
                new_min = order[pos]
        
            # Since a node needs to contain other nodes in its quorum it must
            if new_min > min_quorums[ni]:
                min_quorums[ni] = new_min
                changed = True
                break

    len_good = len(set(node_q) - bad_nodes)
    all_good = 0
    for ni in set(node_q) - bad_nodes:
        print(ni, min_quorums[ni], int(math.floor(len_good/2)))
        if min_quorums[ni] > int(math.floor(len_good/2)):
            all_good += 1 

    return all_good, ref_quorums, min_quorums, just_good_nodes


def stats_cutoff(Scored_nodes, G, cut_off, Sybil_nodes):
    Sybil_nodes = set(Sybil_nodes)
    orig_good = set([n for n,_, y in Scored_nodes if y >= cut_off])
    orig_bad = set([n for n,_, y in Scored_nodes if y < cut_off])

    for n in Sybil_nodes:
        assert n in G.nodes()

    fpos = len([n for n in orig_good if n in Sybil_nodes]) / (len(orig_good) + 0.000001)
    fneg = len([n for n in orig_bad if n not in Sybil_nodes]) / (len(orig_bad) + 0.000001)

    in_good, in_bad, in_between = 0, 0, 0
    for (u, v) in G.edges:
        if u in orig_good and v in orig_good:
            in_good += 1
        elif u in orig_bad and v in orig_bad:
            in_bad += 1
        else:
            in_between += 1 

    return (cut_off, len(orig_good), len(orig_bad), in_good, len(orig_good), in_bad, len(orig_bad), in_between, fpos, fneg)

def experiment(G, node_sets, Node_samples = 100, k = 1):
    (Good_nodes, Naive_nodes, Sybil_nodes) = node_sets
    num_naive = len(Naive_nodes)

    Total_stake, Stake_all, _ = state_stats(G, node_sets)
    target_stake, target_dist = compute_targets(Stake_all, Total_stake, node_sets)
    M = compute_matrix(G, node_sets)

    walk_len =  15 * (log(M.shape[0]) / log(3))
    print("Walk length: %s" % int(walk_len))
    
    goodlen = len(Good_nodes)
    Good_set = set(Good_nodes)

    all_nodes = Good_nodes + Sybil_nodes

    Bad_samples = int(math.ceil(Node_samples * (len(all_nodes) - goodlen) / len(Good_nodes))) - 1
    print("Initial honest: %s" % Node_samples)
    targets = random.sample(list(range(goodlen)), Node_samples)
    target_sybils = random.sample(list(range(goodlen, len(all_nodes))), Bad_samples  )
    
    print("Logistic: k: %d" % k)
    Dists = sybil_weights(targets, M, target_dist, Total_stake, walk_len, k=k)
    
    # Run an analysis to find out who cannot make a good quorum.
    node_q = {}
    good_nodes = set(targets)
    bad_nodes = set(target_sybils)

    for n in good_nodes:
        assert all_nodes[n] in Good_nodes
    for n in bad_nodes:
        assert all_nodes[n] in Sybil_nodes    
    
    # Do a connectivity analysis from the point of view of a good, non-naive node
    for target_0, Y_0 in Dists:
        if all_nodes[target_0] not in Naive_nodes:
            assert all_nodes[target_0] in Good_nodes
            print("Selected target: %d" % target_0)
            break

    Scored_nodes = list(zip(all_nodes, range(len(Y_0)), Y_0))

    actual_cut_off = 0.0
    #prev_in_between = None
    #for cf in np.arange(0.49, 0.51, 0.001):
    #    stats = stats_cutoff(Scored_nodes, G, cut_off=cf, Sybil_nodes=Sybil_nodes)
    #    _, lgood, lbad, in_good, orig_good, in_bad, orig_bad, in_between, fp, fn = stats
    #
    #    dnodes = round(lgood/(lbad+0.000001),4)
    #    dlinks = round( 2*in_good/(in_between + 0.000001) ,4)
    #
    #    # Define condition:
    #    flag = " "
    #    if float(in_good) > float(in_between)/2 and dnodes - dlinks < 0:
    #        flag = "*"
    #        if actual_cut_off == 0.0:
    #            actual_cut_off = cf
    #            
    #   
    #    print("%2.3f\t%d\t%d\t%d\t%d\t%2.2f\t%2.2f\t%s" % (cf, lgood, lbad, in_good, in_between, round(fp,3), round(fn,3), flag))
    #    print(dnodes, dlinks, round(dnodes - dlinks, 2))

    stats = stats_cutoff(Scored_nodes, G, cut_off=0.5, Sybil_nodes=Sybil_nodes)
    _, lgood, lbad, in_good, orig_good, in_bad, orig_bad, in_between, fp, fn = stats
    dnodes = round(lgood/(lbad+0.000001),4)
    dlinks = round( 2*in_good/(in_between + 0.000001) ,4)

    if float(in_good) > float(in_between)/2 and dnodes - dlinks < 1.0:
        actual_cut_off = 0.5
        flag = "*"
    else:
        actual_cut_off = 0.0
        flag = " "

    print("%2.3f\t%d\t%d\t%d\t%d\t%2.2f\t%2.2f\t%s" % (0.5, lgood, lbad, in_good, in_between, round(fp,3), round(fn,3), flag))
    print(dnodes, dlinks, round(dnodes - dlinks, 2))


    cut_off = actual_cut_off
    print("Cutoff: %2.3f" % cut_off)

    for target, Y in Dists:
        node_q[target] = set()

        for i in targets + target_sybils:
            y = Y[i]
            assert i in bad_nodes or i in good_nodes
            # print(i, y, ["", "*"][bool(y >= cut_off)])
            if y >= cut_off:
                node_q[target].add( i )
            
        n_good = len([n for n in node_q[target] if n in good_nodes])
        for n in [n for n in node_q[target] if n in good_nodes]:
            assert n in good_nodes

        n_all = len(node_q[target])
        # print( "%d\t%d\t%d\t%2.2f" % (target, n_good, n_all, n_good / n_all))
        # print(node_q[target])
        

    new_bad_nodes = bad_nodes | set()
    while len(new_bad_nodes) > 0:
        old_bad_nodes = new_bad_nodes
        new_bad_nodes = set()

        for ni in set(node_q) - bad_nodes:
            if not len(node_q[ni] - bad_nodes) > 2 * len(node_q[ni] & bad_nodes):
                bad_nodes.add(ni)
                new_bad_nodes.add(ni)

    all_good, _, _, just_good_nodes = test_all_quorums_for_size(node_q, bad_nodes)
    
    out_good = len(just_good_nodes)
    out_confused = (Node_samples - len(just_good_nodes))

    print("  --- Results ---")
    print("   Pure Good: %d" % out_good)
    print("    Confused: %d" % out_confused    )
    print("      Sybils: %d" % Bad_samples)
    print("In agreement: %d" % (all_good))
    print("  ---------------")

    return out_good, out_confused, Bad_samples, all_good

def experiment_benign():
    G = nx.read_edgelist("samples/small_net1.txt", nodetype=int, create_using=nx.DiGraph())

    deg = G.out_degree()
    Good_stake = sum(v for _, v in deg)
    print("Good Stake: %s" % Good_stake)


    none = {
        "num_sybils"  : 1 , 
        "stake_sybils": 0 , 
        "frac_naive"  : 0.5, 
        "stake_naive" : 6 
    }

    byzantine = {
        "num_sybils"  : int(math.ceil(len(G.nodes)/2)) - 1 , 
        "stake_sybils": int(0.50 * Good_stake / 2) - 1, 
        "frac_naive"  : 1.0, 
        "stake_naive" : int(0.50 * Good_stake / 2 ) - 1 
    }


    seed = int(1000000 * random.random())
    print("Random seed: %s" % seed)
    random.seed(seed)

    for i in range(10):
        print("Condition: none")
        Gx, node_sets = connect_sybils(G, **none)
        experiment(Gx, node_sets)


    for i in range(10):
        print("Condition: byzantine")
        Gx, node_sets = connect_sybils(G, **byzantine)
        experiment(Gx, node_sets)


def main():
    G = nx.read_edgelist("samples/small_net1.txt", nodetype=int, create_using=nx.DiGraph())

    deg = G.out_degree()
    Good_stake = sum(v for _, v in deg)
    print("Good Stake: %s" % Good_stake)


    none = {
        "num_sybils"  : 1 , 
        "stake_sybils": 0 , 
        "frac_naive"  : 0.5, 
        "stake_naive" : 6 
    }

    byzantine = {
        "num_sybils"  : int(math.ceil(len(G.nodes)/2)) - 1 , 
        "stake_sybils": int(0.50 * Good_stake / 2) - 1, 
        "frac_naive"  : 1.0, 
        "stake_naive" : int(0.50 * Good_stake / 2 ) - 1 
    }

    normal = {
        "num_sybils"  : len(G.nodes), 
        "stake_sybils": int(0.85 * Good_stake) , 
        "frac_naive"  : 0.3, 
        "stake_naive" : int(0.15 * Good_stake) 
    }

    normal_bad = {
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

    Gx, node_sets = connect_sybils(G, **normal)
    experiment(Gx, node_sets)
    return

    seed = int(1000000 * random.random())
    print("Random seed: %s" % seed)
    random.seed(seed)

    for v in np.arange(0.5, 10.0, 0.5):
        normal["stake_sybils"] = int(Good_stake * v)
 
        for i in range(5):
            print("Condition: none")
            Gx, node_sets = connect_sybils(G, **normal)
            experiment(Gx, node_sets)


if __name__ == "__main__":
    main()