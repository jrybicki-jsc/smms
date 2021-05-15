# spliter
import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import turicreate as tc
import argparse
import pathlib
import datetime
import os
import datetime
from grapm import f_create_network, convert_to_voting, save_nets
import logging
from streamed import get_anchor_coords
from utils import setup_logging, setup_path, load_functions_partition


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Calculate network for a partitions')
    parser.add_argument('--functions', help='name of the functions directory', required=True)
    parser.add_argument('--p', help='partition number', type=int, required=True )
    parser.add_argument('--output', help='output path', required=True)
    parser.add_argument('--gamma', help='gamma', default=.65, type=float)
    args = parser.parse_args()

    path = setup_path(args=args)
    setup_logging(path=path, parser=parser)


    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY',5*2147483648)
    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE', 5*134217728)
    # following can reduce the memory footprint
    tc.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)

    mw = load_functions_partition(directory=args.functions, name=args.p)

    gamma = args.gamma 
    if gamma > 1.0:
        gamma = gamma/10.0

    logging.info(f"Stargng network calculation for gamma={gamma}")
    
    
    net = f_create_network(data=mw, gamma=gamma)
    save_nets({args.gamma: [net]}, f"{args.gamma}-{args.p}-tc-nets",  directory=path)
    logging.info(f"Network with {len(net)} anchors saved ")

    anchors = get_anchor_coords(net=net, data=mw)
    pp = os.path.join(path, f"anchors-{args.p}")
    anchors.save(pp, format='binary')
    logging.info(f"Anchor cords saved in {pp}")

