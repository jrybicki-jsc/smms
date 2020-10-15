from appknn import adf, create_aggregating_net, app_k_nearest, partition_data, mysample
import pandas as pd
import numpy as np
from multiprocessing import Pool
from collections import defaultdict
from tqdm import tqdm
import pickle
from functools import partial
import time


def save_nets(nets, name):
    normal_nets = dict()
    for gamma, [n1, n2] in nets.items():
        normal_nets[gamma] = [dict(n1), dict(n2)]

    with open(f"res/{name}.pickle", 'wb+') as f:
        pickle.dump(normal_nets, f)


def partition_dataframe(df, n_parts):
    permuted_indices = np.random.permutation(len(df))

    dfs = []
    for i in range(n_parts):
        dfs.append(df.iloc[permuted_indices[i::n_parts]])
    return dfs


def find_closest(net, app, distance, k=1):
    anchors = app_k_nearest(k=k, apps=net.keys(), new_app=app, distance=distance)
    new_list = []
    for a in anchors:
        if len(net[a])> 0:
            new_list+=net[a]
        new_list.append(a)       
    ns=app_k_nearest(k=1, apps=new_list, new_app=app, distance=distance)[0]
    return ns

def classify_app(appid, labels):
    return labels[labels.apn==appid]['malware_label'].values[0]


def votes_from_list(li, labels):
    classes = [int(classify_app(a, labels)) for a in li]
    su = sum(classes)
    return len(classes)-su, su

def convert_to_voting(net, labels):
    voting_network = defaultdict(lambda: [0, 0])

    for k, l in net.items():
        bi, mal = votes_from_list(l+[k], labels)
        voting_network[k]=[bi, mal]
        
    return voting_network

def merge_voting_nets(nets, distance, gamma):
    apns = []
    for net in nets:
        apns+=list(net.keys())
        
    nn = create_aggregating_net(gamma=gamma,
                                apns=apns, 
                                distance=distance)
    targ = defaultdict(lambda: [0, 0])

    for k,v in nn.items():
        for net in nets:
            targ[k] = [i+j for i,j in zip(targ[k], net.get(k, [0, 0]))]
            
        for el in v:
            for net in nets:
                targ[k] = [i+j for i,j in zip(targ[k], net.get(el, [0, 0]))]
            #targ[k] = [i+j for i,j in zip(targ[k], convert_to_voting([el], labels))]
    
    return targ

def rr(x, gamma):
    print(f"Doing some work...{gamma=} {len(x)}")
    net = create_aggregating_net(
        gamma=gamma, apns=x.keys(), distance=lambda x1, y1: adf(x1, y1, x))
    #vot_net = convert_to_voting(net, labels)
    return net

def make_and_merge(funcs, labels, gamma):
    myfc = partial(rr, gamma=gamma)
    parts = partition_dataframe(funcs, 10)
    with Pool() as p:
            networks = p.map(myfc, parts)

    print("Converting...")
    start = time.time()
    voting_networks = [convert_to_voting(net, labels) for net in networks]
    end = time.time()
    print(f"\tElapsed: {end-start}")

    print("Merging")
    start = time.time()
    mv = merge_voting_nets(nets=voting_networks, distance=lambda x,y: adf(x,y, funcs), gamma=gamma)
    end = time.time()
    print(f"\tElapsed: {end-start}")


def exp(funcs):
    res = dict()
    res_ref = dict()
    nets = dict()
    gamma = 0

    parts = partition_dataframe(funcs, 10)


    for gamma in tqdm([0, 1, 2, 4, 8, 16, 32, 64, 72, 80, 88, 96, 104, 110, 128, 164, 256]):
        print(f"Current {gamma=}")
        
            #[create_aggregating_net(gamma=gamma, apns=part, distance=distance) for part in partitions]
        voting_networks = [convert_to_voting(net, labels) for net in networks]
        mv = merge_voting_nets(nets=voting_networks, distance=distance, gamma=gamma)
        
        reference_netw = create_aggregating_net(gamma=gamma, apns=smp.apn.unique(), distance=lambda x,y: adf(x,y, funcs_smp))
        reference_voting = convert_to_voting(reference_netw, labels)
        
        #save the net
        nets[gamma] = [mv.copy(), reference_voting.copy()]
        false_negative, false_positives = evaluate_voting_net(apns=apns, net=mv)
        res[gamma] = [false_negative, false_positives]
        
        false_negative, false_positives = evaluate_voting_net(apns=apns, net=reference_voting)
        res_ref[gamma] = [false_negative, false_positives]
        
        if gamma ==0:
            gamma=1
        else:
            gamma*=2
        
        print(f"Anchor points: {len(reference_voting.keys())}")
        if len(reference_voting.keys()) == 1:
            break
            
    save_nets(nets=nets, name=f"{sample_size}-votingnets")

if __name__ == "__main__":
    v = pd.read_csv('data/functions_encoded.csv')
    funcs = v.groupby(by='apn')['nf'].apply(set)
    labels = pd.read_csv('data/labels_encoded.csv')

    #sample_size = 200
    #smp = mysample(v, sample_size)
    #funcs_smp = smp.groupby(by='apn')['nf'].apply(set)

    make_and_merge(funcs=funcs, labels=labels, gamma=4)