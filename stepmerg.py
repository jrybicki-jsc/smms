from mulp import evaluate_voting_net
from tqdm import tqdm
from appknn import app_k_nearest, adf, create_aggregating_net, lcl
import pandas as pd
import numpy as np
import pickle
from functools import partial
from multiprocessing import Pool
from merging import merge_voting_nets


def ge_anchors(gamma):
    with open(f"./res/{gamma}-singlevoting.pickle", 'rb') as f:
        nets = pickle.load(f)

    anchors = list()

    for n in nets[gamma]:
        anchors = anchors + list(n.keys())
    return anchors, nets


def generate_merged(gamma, distance, labels):
    _, nets = ge_anchors(gamma)
    nets = nets[gamma]

    print(f"One shot merging of {len(nets)} for {gamma=}")
    onsm = merge_voting_nets(nets=nets, distance=distance, gamma=gamma)

    print("Pariwise merging")

    part_merge = partial(merge_voting_nets, gamma=gamma, distance=distance)
    while len(nets) > 1:
        print(f"Hierarchical merging {len(nets)}")
        b = zip(nets[::2], nets[1::2])
        with Pool() as p:
            nets = p.map(part_merge, b)

    return onsm, nets[0]


if __name__ == "__main__":
    print("reading labels")
    labels = pd.read_csv('data/labels_encoded.csv')

    print("reading functions")
    v = pd.read_csv('data/functions_encoded.csv')
    funcs = v.groupby(by='apn')['nf'].apply(set)
    def distance(x, y): return adf(x, y, funcs)
    def classifier(x): return lcl(x, labels)

    print('reading test ste')
    tests = pd.read_csv('res2/9500-test.csv', index_col=0)

    res = dict()
    sizes = dict()

    for gamma in tqdm([0, 1, 2, 4, 8, 16, 32, 180, 192]):
        onsm, nets = generate_merged(gamma=gamma, distance=distance, labels=labels)
        false_negative, false_positives = evaluate_voting_net(
            apns=tests.index,
            net=onsm, 
            distance=distance,
            classifier=classifier)

        false_negative_steps, false_positives_steps = evaluate_voting_net(
            apns=tests.index, 
            net=nets, 
            distance=distance,
            classifier=classifier)

        res[gamma] = [false_negative, false_positives,
                      false_negative_steps, false_positives_steps]
        sizes[gamma] = [len(onsm.keys()), len(nets.keys())]

        with open(f"res/mergers-{gamma}.pickle", 'wb+') as f:
            pickle.dump([dict(onsm), dict(nets)], f)

    merged = pd.DataFrame.from_dict(res, orient='index', columns=['one_fPos', 'one_fNeg', 'steps_fPos', 'steps_fNeg'])
    merged.to_csv(f"res/{len(tests)}-mergecomparision.csv")

    sizes = pd.DataFrame.from_dict(sizes, orient='index', columns=['one_anchors',  'steps_anchors'])
    sizes.to_csv(f"res/{len(tests)}-mergecomparision-anchors.csv")
