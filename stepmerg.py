from mulp import evaluate_voting_net
from tqdm import tqdm
from appknn import app_k_nearest, adf, create_aggregating_net, lcl, jaccard
import pandas as pd
import numpy as np
import pickle
from functools import partial
from multiprocessing import Pool
from merging import net_based_multi_merge
from sklearn.model_selection import train_test_split
import time
from dataprep import mysample, partition_dataframe


def ge_anchors(gamma):
    with open(f"./res/{gamma}-singlevoting.pickle", 'rb') as f:
        
        nets = pickle.load(f)

    anchors = list()

    for n in nets[gamma]:
        anchors = anchors + list(n.keys())
    return anchors, nets

def save_nets(nets, name):

    with open(f"res/{name}.pickle", 'wb+') as f:
        pickle.dump(nets, f)

def distance(x, y, arg):
    return jaccard(x, y, arg)

def single_net(x, gamma, classifier):
    start = time.time()
    net = create_aggregating_net(gamma=gamma, apns=x.keys(), distance=lambda x1, y1: distance(x1, y1, x))
    end = time.time()
    print(f"Creating agg network...{gamma=} {len(x)} Elapsed: {end-start}")
    return net

def generate_merged(gamma, distance):

    myfc = partial(single_net, gamma=gamma)
    parts = partition_dataframe(funcs, 8)
    with Pool() as p:
        agg_networks = p.map(myfc, parts)
    
    save_nets({gamma: agg_networks}, f"{gamma}-singleaggregating")

    
    print("Pairwise merging")
    nets = agg_networks
    part_merge = partial(net_based_multi_merge, gamma=gamma, distance=distance)
    while len(nets) > 1:
        print(f"Hierarchical merging {len(nets)}")
        b = zip(nets[::2], nets[1::2])
        with Pool() as p:
            nets = p.map(part_merge, b)

    return nets[0]


if __name__ == "__main__":
    print("reading labels")
    labels = pd.read_csv('data/labels_encoded.csv')

    print("reading functions")
    v = pd.read_csv('data/functions_encoded.csv')
    funcs = v.groupby(by='apn')['nf'].apply(set)
    d = lambda x, y: distance(x, y, funcs)

    #print('reading test ste')
    #tests = pd.read_csv('res2/9500-test.csv', index_col=0)
    test_size = 200
    train, test = train_test_split(funcs, test_size=test_size, random_state=42)
    test.to_csv(f"/res/test-{test_size}.csv")


    #for gamma in tqdm([0, 1, 2, 4, 8, 16, 32, 180, 192]):
    #for gamma in tqdm([0, 0.1, 0.4, 0.5, 0.7, 0.8, 0.85, 0.9, 1.0]):
    intervals = 18
    for gamma in tqdm([x * 1/intervals for x in range(0, intervals+1)]):
        #onsm, nets = generate_merged(gamma=gamma, distance=distance, labels=labels)
        print("Creating reference voting netwrok")
        start = time.time()
        reference = create_aggregating_net(gamma=gamma, apns=train.index, distance=d)
        end = time.time()
        print(f"\tElapsed: {end-start}")

        merged = generate_merged(gamma=gamma, distance=d)

        
        with open(f"res/mergers-jaccard-{gamma}.pickle", 'wb+') as f:
            pickle.dump([dict(reference), dict(merged)], f)

    