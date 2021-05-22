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
                   save_nets, get_sample)

from streamed import f_create_network, get_anchor_coords, tc_based_nn, naive_merge

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
#    for gamma in tqdm([x * 1/intervals for x in range(0, intervals+1)]):
    gamma = 0.65
    print(f"Current {gamma=}")
   
    for p in range(0, len(parts)):
        print(f"Creating origin network {p=}")
        origin_net = f_create_network(data=sparts[p], gamma=gamma)
        origin_anchors = get_anchor_coords(net=origin_net, data=sparts[p])

        print('Creating streamed networks')
        start = time.time()
        neigh = [tc_based_nn(net=origin_net, apks=list(par), data=origin_anchors.append(spar)) 
            for par, spar in zip(parts[0:p]+parts[p+1:len(parts)], sparts[0:p]+sparts[p+1:len(sparts)])]
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
        nets[gamma] = voting
        # save nets:
        save_nets(nets=nets, name=f"{len(train)}-{p}-stable-stream-nets")
