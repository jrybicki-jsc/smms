#naive merge nets
import argparse
import logging
import os
import pickle

import turicreate as tc
from streamed import naive_merge
from utils import (load_functions_partition, setup_logging, setup_path,
                   setup_turi)
from grapm import save_nets


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Naive merge networks')
    parser.add_argument('--nets', help='name of the network directory', required=True)
    parser.add_argument('--p1', help='partition number', type=int, required=True)
    parser.add_argument('--p2', help='partition number', type=int, required=True)
    parser.add_argument('--origin', help='index of origin network', type=int, required=True, default=0)
    parser.add_argument('--output', help='output path', required=True)
    parser.add_argument('--gamma', help='gamma', default=.65, type=float)
    args = parser.parse_args()

    path = setup_path(args=args)
    setup_logging(path=path, parser=parser)
    setup_turi()

    gamma = args.gamma
    if gamma > 10:
        gamma = gamma/10.0

    if gamma > 1.0:
        gamma = gamma/10.0

    
    logging.info(f"Loading networks {gamma}")
    networks = list()
    for i in range(args.p1, args.p2+1):
        with open(os.path.join(args.nets, f"{gamma}-streamed-{i}.pickle"), 'rb') as f:
            net = pickle.load(f)
        networks.append(list(net.values())[0][0])
        g2 = list(net.keys())[0]
        if g2!=gamma:
            logging.warning(f"Found different gamman in network file {i}: {gamma}!={g2}")
            gamma =g2 

    origin_net = networks[args.origin]
    del networks[args.origin]

    logging.info(f"Starting to naive merge {len(networks)} nets with gamma={gamma}")
    merged = origin_net.copy()
    for d in networks:
        merged = naive_merge(merged, d)
    
    save_nets({gamma: [merged]}, f"merged-{gamma}-{args.origin}-tc-nets",  directory=path)
    logging.info(f"Saved network with {len(merged)}")
