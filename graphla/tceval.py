#!/usr/bin/env python

import pickle
import pandas as pd
from tqdm import tqdm
import argparse
import turicreate as tc
from streamed import tc_based_nn
import numpy as np


def s_conver_to_probs(v):
    return 1.0 - v[1]/(v[0]+v[1])

def eval_net(net, test_apns, data):
    nns = tc_based_nn(net, list(test_apns), data)
    return [s_conver_to_probs(net[row['nn']]) for row in nns.sort('apk')]
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process networks.')
    parser.add_argument('--net-file', help='name of the network file', required=True)
    parser.add_argument('--test-file', help='name of the test file', required=True)
    parser.add_argument('--out', help='name of the output', default='res/out.pickle')
    args = parser.parse_args()
    print(args)

    with open(args.net_file, 'rb') as f:
        nets = pickle.load(f)

    test_apns = np.load(args.test_file)

    test_apns.sort()
    labels = pd.read_csv('../data/labels_encoded.csv', index_col=0)
    true_values = [not labels.loc[a]['malware_label'] for a in test_apns]

    mw = tc.load_sframe('../binarydata/funcs-encoded')
    mw = mw.remove_column('fcount', inplace=True)

    merges = dict()
    refs = dict()
    for gamma, net in tqdm(nets.items()):
        refs[gamma] = [eval_net(net=net, test_apns=test_apns, data=mw), true_values]
        
    with open(args.out, 'wb+') as f:
        pickle.dump(refs, f)

