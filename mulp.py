import pickle
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from appknn import (adf, app_k_nearest, create_aggregating_net, mysample,
                    partition_data)


def save_nets(nets, name):
    #normal_nets = dict()
    # for gamma, [n1, n2] in nets.items():
    #    normal_nets[gamma] = [dict(n1), dict(n2)]

    with open(f"res/{name}.pickle", 'wb+') as f:
        pickle.dump(nets, f)


def partition_dataframe(df, n_parts):
    rn = default_rng(42)
    permuted_indices = rn.permutation(len(df))

    dfs = []
    for i in range(n_parts):
        dfs.append(df.iloc[permuted_indices[i::n_parts]])
    return dfs


def find_closest(net, app, distance, k=1):
    anchors = app_k_nearest(k=k, apps=net.keys(),
                            new_app=app, distance=distance)
    new_list = []
    for a in anchors:
        if len(net[a]) > 0:
            new_list += net[a]
        new_list.append(a)
    ns = app_k_nearest(k=1, apps=new_list, new_app=app, distance=distance)[0]
    return ns


def classify_app(appid, labels):
    return labels[labels.apn == appid]['malware_label'].values[0]


def votes_from_list(li, labels):
    classes = [int(classify_app(a, labels)) for a in li]
    su = sum(classes)
    return len(classes)-su, su


def convert_to_voting(net, labels):
    voting_network = defaultdict(lambda: [0, 0])

    for k, l in net.items():
        bi, mal = votes_from_list(l+[k], labels)
        voting_network[k] = [bi, mal]

    return voting_network


def merge_voting_nets(nets, distance, gamma):
    apns = []
    for net in nets:
        apns += list(net.keys())

    nn = create_aggregating_net(gamma=gamma,
                                apns=apns,
                                distance=distance)
    targ = defaultdict(lambda: [0, 0])

    for k, v in nn.items():
        for net in nets:
            targ[k] = [i+j for i, j in zip(targ[k], net.get(k, [0, 0]))]

        for el in v:
            for net in nets:
                targ[k] = [i+j for i, j in zip(targ[k], net.get(el, [0, 0]))]
            #targ[k] = [i+j for i,j in zip(targ[k], convert_to_voting([el], labels))]

    return targ


def classify_using_voting(app, net, distance, k=1):
    ns = app_k_nearest(k=k, apps=net.keys(), new_app=app, distance=distance)

    ret = [0, 0]
    for n in ns:
        ret = np.add(ret, net[n])
    return ret


def evaluate_voting_net(apns, net, distance, k=3):
    fp = 0
    fn = 0
    for a in apns:
        v1 = classify_using_voting(app=a, net=net, distance=distance, k=k)

        gt = classify_app(a, labels)
        rt = v1[0] < v1[1]
        if gt != rt:
            if rt == True:
                fp += 1
            else:
                fn += 1

    return fn, fp


def rr(x, gamma):
    start = time.time()
    print(f"Creating agg network...{gamma=} {len(x)}")
    net = create_aggregating_net(
        gamma=gamma, apns=x.keys(), distance=lambda x1, y1: adf(x1, y1, x))
    #vot_net = convert_to_voting(net, labels)
    end = time.time()
    print(f"\tElapsed: {end-start}")
    return net


def mymerge(pair, gamma):
    ((n1, p1), (n2, p2)) = pair
    funcs = p1.append(p2)

    print(f"Merging {len(funcs)}")
    n = merge_voting_nets(
        nets=[n1, n2], distance=lambda x, y: adf(x, y, funcs), gamma=gamma)
    return (dict(n), funcs)


def make_and_merge(funcs, labels, gamma):
    myfc = partial(rr, gamma=gamma)
    parts = partition_dataframe(funcs, 8)
    with Pool() as p:
        networks = p.map(myfc, parts)

    print("Converting...")
    start = time.time()
    voting_networks = [dict(convert_to_voting(net, labels))
                       for net in networks]
    end = time.time()
    print(f"\tElapsed: {end-start}")

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


def myf(x):
    (a, b) = x
    print(f"{a=}\n{b=}\n")


def exp(funcs, labels, test_size=10):
    res = dict()
    res_ref = dict()
    nets = dict()
    gamma = 0

    train, test = train_test_split(funcs, test_size=test_size, random_state=42)

    for gamma in tqdm([0, 1, 2, 4, 8, 16, 32, 64, 72, 80, 88, 96, 104, 110, 128, 164, 180, 192]):
        print(f"Current {gamma=}")
        mv = make_and_merge(train, labels, gamma)

        false_negative, false_positives = evaluate_voting_net(apns=test.index, net=mv, distance=lambda x, y: adf(x, y, funcs))
        res[gamma] = [false_negative, false_positives]

        print("Creating and converting reference netwrok")
        start = time.time()
        reference_netw = create_aggregating_net(gamma=gamma, apns=train.index, distance=lambda x, y: adf(x, y, funcs))
        reference_voting = convert_to_voting(reference_netw, labels)
        end = time.time()
        print(f"\tElapsed: {end-start}")

        false_negative, false_positives = evaluate_voting_net(apns=test.index, net=reference_voting, distance=lambda x, y: adf(x, y, funcs))
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
    labels = pd.read_csv('data/labels_encoded.csv')

    sample_size = 200
    smp = mysample(v, sample_size)
    funcs_smp = smp.groupby(by='apn')['nf'].apply(set)

    exp(funcs_smp, labels, 50)
