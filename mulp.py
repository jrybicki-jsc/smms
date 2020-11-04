import pickle
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from merging import merge_voting_nets

import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from appknn import (adf, create_aggregating_net, create_voting_net, evaluate_voting_net, lcl)
from dataprep import mysample, partition_dataframe


def save_nets(nets, name):
    #normal_nets = dict()
    # for gamma, [n1, n2] in nets.items():
    #    normal_nets[gamma] = [dict(n1), dict(n2)]

    with open(f"res/{name}.pickle", 'wb+') as f:
        pickle.dump(nets, f)

def mymerge(pair, gamma):
    ((n1, p1), (n2, p2)) = pair
    funcs = p1.append(p2)

    print(f"Merging {len(funcs)}")
    n = merge_voting_nets(
        nets=[n1, n2], distance=lambda x, y: adf(x, y, funcs), gamma=gamma)
    return (dict(n), funcs)


def rr(x, gamma, classifier):
    start = time.time()
    #net = create_aggregating_net(
    #    gamma=gamma, apns=x.keys(), distance=lambda x1, y1: adf(x1, y1, x))
    net = create_voting_net(
            gamma=gamma, apns=x.keys(), distance=lambda x1, y1: adf(x1, y1, x), 
            classifier=classifier)
    end = time.time()
    print(f"Creating voting network...{gamma=} {len(x)} Elapsed: {end-start}")
    return net

def myclass(x, labels):
    return [[0, 1], [1, 0]][int(labels.loc[x]['malware_label'])]

def make_and_merge(funcs, labels, gamma):
    print(f"Starting network creation {gamma}")
    classifier =  partial(myclass, labels = labels) 
    #lambda x: [[0, 1], [1, 0]][int(labels.loc[x]['malware_label'])]
    myfc = partial(rr, gamma=gamma, classifier=classifier)
    parts = partition_dataframe(funcs, 8)
    with Pool() as p:
        voting_networks = p.map(myfc, parts)

    save_nets({gamma: voting_networks}, f"{gamma}-singlevoting")

    print("Merging")
    start = time.time()

    #nds = list(zip(voting_networks, parts))
    #part_merge = partial(mymerge, gamma=gamma)
    # while len(nds)>1:
    #    print(f"Hierarchical merging {len(nds)}")
    #    b = zip(nds[::2], nds[1::2])
    #    with Pool() as p:
    #        nds = p.map(part_merge,b)
    mv = merge_voting_nets(nets=voting_networks,
                           distance=lambda x, y: adf(x, y, funcs), gamma=gamma)
    end = time.time()
    print(f"\tElapsed: {end-start}")
    return mv


def exp(funcs, labels, test_size=10):
    res = dict()
    res_ref = dict()
    nets = dict()
    gamma = 0

    train, test = train_test_split(funcs, test_size=test_size, random_state=42)
    test.to_csv('/tmp/test.csv')
    classifier = lambda x: [[0, 1], [1, 0]][int(labels.loc[x]['malware_label'])]

    for gamma in tqdm([0, 1, 2, 4, 8, 16, 32, 64, 72, 80, 88, 96, 104, 110, 128, 164, 180, 192]):
        print(f"Current {gamma=}")
        mv = make_and_merge(train, labels, gamma)

        false_negative, false_positives = evaluate_voting_net(
            apns=test.index,
            net=mv, 
            distance=lambda x, y: adf(x, y, funcs),
            classifier=classifier)
        res[gamma] = [false_negative, false_positives]

        print("Creating reference voting netwrok")
        start = time.time()
        #reference_netw = create_aggregating_net(gamma=gamma, apns=train.index, distance=lambda x, y: adf(x, y, funcs))
        reference_voting = create_voting_net(
            gamma=gamma, apns=train.index, distance=lambda x, y: adf(x, y, funcs), classifier=classifier)
        end = time.time()
        print(f"\tElapsed: {end-start}")

        false_negative, false_positives = evaluate_voting_net(
            apns=test.index,
            net=reference_voting,
            distance=lambda x, y: adf(x, y, funcs),
            classifier=classifier)
        res_ref[gamma] = [false_negative, false_positives]

        nets[gamma] = [dict(mv), dict(reference_voting)]
        if gamma == 0:
            gamma = 1
        else:
            gamma *= 2

        print(f"Anchor points: {len(mv.keys())}")
        if len(mv.keys()) == 1:
            break

    # save results:
    save_nets(nets=nets, name=f"{len(train)}-votingnets")


    merged = pd.DataFrame.from_dict({gamma: res[gamma] + res_ref[gamma]  for gamma in res.keys()}, 
             orient='index', columns=['mer_fPos', 'mer_fNeg', 'ref_fPos', 'ref_fNeg'])
    merged.to_csv(f"res/{len(train)}-mergedresults.csv")


if __name__ == "__main__":
    v = pd.read_csv('data/functions_encoded.csv')
    funcs = v.groupby(by='apn')['nf'].apply(set)
    labels = pd.read_csv('data/labels_encoded.csv', index_col=0)

    sample_size = 200
    smp = mysample(v, sample_size)
    funcs_smp = smp.groupby(by='apn')['nf'].apply(set)

    exp(funcs_smp, labels, 50)
