import numpy as np
from typing import Tuple, Sequence, Callable
import random
import itertools
from collections import defaultdict
from numpy.random import default_rng
from dataprep import mysample

def adf(apid1: int, apid2: int, funcs) -> float:
    p1 = funcs[apid1]
    p2 = funcs[apid2]

    a = len(p1.difference(p2)) + len(p2.difference(p1))
    return np.sqrt(a)

def jaccard(apid1: int, apid2: int, funcs) -> float:
    p1 = funcs[apid1]
    p2 = funcs[apid2]

    return 1 - len(p1 & p2)/len(p1|p2) 


def create_net(gamma: float, apns: Sequence[int], distance: Callable) -> Sequence[int]:
    np.random.shuffle(apns)
    net = []

    for a in apns:
        insert = True
        for n in net:
            if distance(a, n) < gamma:
                insert = False
                break
        if insert:
            net.append(a)

    return net


def create_aggregating_net(gamma: float, apns: Sequence[int], distance: Callable[[int, int], float]):
    net = dict()

    for a in apns:
        insert = True
        for n in net.keys():
            if distance(a, n) <= gamma:
                insert = False
                net[n].append(a)
                break  
        if insert:
            net[a] = list()

    return net


def app_k_nearest(k: int, apps: Sequence[int], new_app: int, distance: Callable):
    byd = sorted(apps, key=lambda lp: distance(lp, new_app))
    return byd[:k]


# split smp according to label (malicious or not)
def split_mal(smp, labels):
    benid = labels[labels.malware_label == False]['apn'].values
    malid = labels[labels.malware_label == True]['apn'].values

    mals = smp[smp.apn.isin(malid)]
    bi = smp[smp.apn.isin(benid)]

    return (mals, bi)


# minimal distance between differently labelled data points (returns distance, and found inconsitent points)
def calculate_margin(smp, labels, distance: Callable) -> Tuple[float, Sequence[Tuple[int, int]]]:
    mals, beni = split_mal(smp, labels)
    print(
        f"Split finished: {mals.shape[0]} malicious, {beni.shape[0]} bening, {smp.shape[0]} overall")
    min_dist = float('inf')
    problematic = []

    for o in beni.apn.unique():
        for z in mals.apn.unique():
            if (d:=distance(o, z, smp)) < min_dist:
                if d == 0:
                    # incosistiencies are ignored
                    problematic.append((o, z))
                else:
                    min_dist = d

    return min_dist, problematic


def margins(v, labels, sample_size, problematic=[]):
    smp = mysample(v, sample_size)  # v.sample(sample_size, random_state=42)
    print(f"sample created {smp.shape[0]/v.shape[0]:.2f}")
    if len(problematic) > 0:
        print(f"Removing problematic {len(problematic)} apps")
        smp = smp[~smp.apn.isin(problematic)]

    funcs_smp = smp.groupby(by='apn')['nf'].apply(set)
    margin, problematic = calculate_margin(
        smp, labels, distance=lambda x, y, z: adf(x, y, funcs_smp))
    return margin, problematic
