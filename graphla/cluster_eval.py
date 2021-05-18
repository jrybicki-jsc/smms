#!/usr/bin/env python
# cluster eval 
import pickle
import pandas as pd 
import logging
import argparse
import turicreate as tc
from streamed import tc_based_nn
import numpy as np
from utils import setup_path, setup_logging
from stream_net import tc_based_nn

def s_conver_to_probs(v):
    return 1.0 - v[1]/(v[0]+v[1])

def eval_net(net, anchors, data):
    #tc_based_nn(net, anchors, partition):
    nns = tc_based_nn(net=net, anchors=anchors, partition=data)
    logging.info('Network done')
    return [(row, s_conver_to_probs(net[row['nn']])) for row in nns.sort('apk')]
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Eval networks.')
    parser.add_argument('--net', help='name of the network file', required=True)
    parser.add_argument('--anchors', help='network anchor file', required=True)
    parser.add_argument('--test-file', help='name of the test file', required=True)
    parser.add_argument('--labels', help='name of the labels file', required=True)
    parser.add_argument('--output', help='name of the output', default='res/out.pickle')
    args = parser.parse_args()

    path = setup_path(args)
    setup_logging(path=path, parser=parser)

    logging.info(f"Loading origin network {args.net} & {args.anchors}")
    with open(args.net, 'rb') as f:
        net = pickle.load(f)
    gamma = list(net.keys())[0]
    net = list(net.values())[0][0]
    an = tc.load_sframe(args.anchors)

    logging.info(f"Reading test file: {args.test_file}")
    test = tc.load_sframe(args.test_file)
    test_apns = list(test['apk'].unique())
    #think about sorting 
    test_apns.sort()

    logging.info(f"Reading labels from {args.labels}")
    labels = pd.read_csv(args.labels, index_col=0)
    classifier = lambda x: int(labels.loc[x]['malware_label'])
    
    logging.info('Setting true values')
    true_values = [not labels.loc[a]['malware_label'] for a in test_apns]

    logging.info('Starting evaluation')
    res= [eval_net(net=net, anchors=an, data=test), true_values]
        
    logging.info('Storing results')
    with open(f"evalresults.pickle", 'wb+') as f:
        pickle.dump(res, f)

