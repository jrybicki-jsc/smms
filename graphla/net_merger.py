#merge nets
import argparse
import datetime
import logging
import os
import pathlib
import pickle

import numpy as np
from tqdm import tqdm

import turicreate as tc
from grapm import f_create_network, save_nets
from utils import (load_functions_partition, setup_logging, setup_path,
                   setup_turi)


def merge_voting_nets(nets, datas, gamma):
    dat = datas[0]
    for data in datas[1:]:
        dat = dat.append(data)

    nn = f_create_network(gamma=gamma, data=dat)
    
    # transfer the "votes" from original networks to just created new anchors
    targ = dict()
    for k, v in nn.items():
        for net in nets:
            if k in net:
                targ[k] = net.get(k)
                break

        for el in v:
            for net in nets:
                if el in net:
                    targ[k] = list(np.add(targ[k], net[el]))
                    break

    return targ



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Merge networks')
    parser.add_argument('--nets', help='name of the network directory', required=True)
    #parser.add_argument('--anchors', hep='name of ')
    parser.add_argument('--p1', help='partition number', type=int, required=True)
    parser.add_argument('--p2', help='partition number', type=int, required=True)
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
    anchors = list()

    for i in range(args.p1, args.p2+1):
        #0.85-0-tc-nets-voting.pickle
        with open(os.path.join(args.nets, f"{gamma}-{i}-tc-nets-voting.pickle"), 'rb') as f:
            net = pickle.load(f)
        networks.append(list(net.values())[0][0])
        g2 = list(net.keys())[0]
        if g2!=gamma:
            logging.warning(f"Found different gamman in network file {i}: {gamma}!={g2}")
            gamma =g2 
        #anchors-0.85-5
        anchorpath = os.path.join(args.nets,f"anchors-{gamma}-{i}")
        an = tc.load_sframe(anchorpath)
        anchors.append(an)

    logging.info(f"Starting to merge {len(networks)} nets with gamma={gamma}")
    r = merge_voting_nets(nets=networks, datas=anchors, gamma=gamma)
    save_nets({gamma: [r]}, f"merged-{gamma}-{args.origin}-tc-nets",  directory=path)
    logging.info(f"Saved network with {len(r)}")
