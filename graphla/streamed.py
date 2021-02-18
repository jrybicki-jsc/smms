import pickle
import time
import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import turicreate as tc
import turicreate.aggregate as agg
from grapm import (convert_to_voting, f_create_network, partition_ndframe,
                   save_nets)


def get_anchor_coords(net, data):
    return data.filter_by(values=list(net.keys()), column_name='apk')

def naive_merge(n1, n2):
    # takes keys from both networks if key overlap the aggregates are merged

    nx = {**n1, **n2}
    for key in nx.keys():
        if key in n1:
            nx[key] = list(set(nx[key]+n1[key]))
    return nx

def naive_voting_merge(n1, n2):
    nx = {**n1, **n2}
    for key in nx.keys():
        if (key in n1) and (key in n2):
            nx[key] = list(np.add(n1[key],nx[key]))
    return nx

def tc_based_nn(net, apks, data):
    anch = list(net.keys())
    anch.extend(apks)
    allt = data.filter_by(values=anch, column_name='apk')

    m = len(anch)
    sim_recom = tc.item_similarity_recommender.create(
        allt, 
        user_id='function', 
        item_id='apk', 
        similarity_type='jaccard', 
        degree_approximation_threshold=15*4096,
        only_top_k=m, verbose=False)
    
    # smaller k could be an optimization here
    items =sim_recom.get_similar_items(apks, k=m)
    # recomendations excluding network anchors 
    fitems = items.filter_by(values=apks, column_name='similar', exclude=True)
    
    return fitems.groupby(key_column_names=['apk'], operations={'nn': tc.aggregate.ARGMAX('score', 'similar')})

if __name__=="__main__":
    mw = tc.load_sframe('../binarydata/funcs-encoded')
    mw.remove_column('fcount', inplace=True)
    #subsamp = get_sample(mw=mw, frac=0.2)
    subsamp = mw

    napks = subsamp['apk'].unique().to_numpy()
    test_size = 1000
    train, test = train_test_split(napks, test_size=test_size, random_state=42)
    np.save(f"../res/test-tc-{test_size}", test)

    parts = partition_ndframe(nd=train, n_parts=4)
    sparts = [subsamp.filter_by(values=part, column_name='apk') for part in parts]
    ftrain = subsamp.filter_by(values=train, column_name='apk')

    labels = pd.read_csv('../data/labels_encoded.csv', index_col=0)
    classifier = lambda x: int(labels.loc[x]['malware_label'])

    nets = dict()
    intervals = 18
    for gamma in tqdm([x * 1/intervals for x in range(0, intervals+1)]):
        print(f"Current {gamma=}")
        print("Creating origin network")
        
        origin_net = f_create_network(data=sparts[0], gamma=gamma)
        origin_anchors = get_anchor_coords(net=origin_net, data=sparts[0])

        print('Creating streamed networks')
        start = time.time()
        neigh = [tc_based_nn(net=origin_net, apks=list(par), data=origin_anchors.append(spar)) for par, spar in zip(parts[1:], sparts[1:])]
        dicts = [net.groupby(key_column_names='nn', operations={'nodes': agg.DISTINCT('apk')}) for net in neigh]
        true_dicts = [{row['nn']: row['nodes'] for row in nep} for nep in dicts]
        end = time.time()
        print(f"\tElapsed: {end-start}")

        print('Merging networks')
        start = time.time()
        merged = origin_net.copy()
        for d in true_dicts:
            merged = naive_merge(merged, d)
        end = time.time()
        print(f"\tElapsed: {end-start}")

        voting = convert_to_voting(merged, classifier)
        
        save_nets({gamma: [true_dicts, origin_net]}, f"{gamma}-stream-singleaggregating")
        nets[gamma] = [merged, voting]
        
        
    # save nets:
    save_nets(nets=nets, name=f"{len(train)}-stream-nets")
