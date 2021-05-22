#!/usr/bin/env python
# cluster eval 
import pickle
import logging
import argparse
import turicreate as tc
from utils import setup_path, setup_logging, load_net
from stream_net import tc_based_nn

def s_conver_to_probs(v):
    return 1.0 - v[1]/(v[0]+v[1])

def eval_net(net, anchors, data):
    #tc_based_nn(net, anchors, partition):
    nns = tc_based_nn(net=net, anchors=anchors, partition=data)
    logging.info('Network done')
    return [(row, s_conver_to_probs(net[row['nn']])) for row in nns.sort('apk')]
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Eval network')
    parser.add_argument('--net', help='name of the network file', required=True)
    parser.add_argument('--anchors', help='network anchor file', required=True)
    parser.add_argument('--test-file', help='name of the test file', required=True)
    parser.add_argument('--output', help='name of the output', default='res/out.pickle')
    args = parser.parse_args()

    path = setup_path(args)
    setup_logging(path=path, parser=parser)

    gamma, net = load_net(args.net)
    an = tc.load_sframe(args.anchors)

    logging.info(f"Reading test file: {args.test_file}")
    test = tc.load_sframe(args.test_file)
    
    logging.info('Starting evaluation')
    res = eval_net(net=net, anchors=an, data=test)
        
    logging.info('Storing results')
    with open(f"{path}/{gamma}-evalresults.pickle", 'wb+') as f:
        pickle.dump(res, f)
