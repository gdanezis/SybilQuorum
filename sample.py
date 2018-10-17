import random


def make_sample_net(num=1):
    f = open("soc-pokec-relationships.txt")
    f2 = open("samples/net%d.txt" % num, "w")

    N = 30622564
    targets = set()
    while len(targets) < 250000:
        v = random.randint(1, N)
        targets.add(v)
            

    for d in f:
        ni, nj = d.strip().split("\t")
        ni, nj = int(ni), int(nj)
        if ni in targets or nj in targets:
            f2.write("%d\t%d\n" % (ni, nj))
            f2.write("%d\t%d\n" % (nj, ni))
            f2.flush()

    f2.close()

for x in range(1, 2):
    make_sample_net(x)