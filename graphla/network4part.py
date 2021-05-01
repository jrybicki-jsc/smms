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
from grapm import partition_ndframe
import datetime
from grapm import f_create_network, convert_to_voting, save_nets
import logging
from streamed import get_anchor_coords


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Calculate network for a partitions')
    parser.add_argument('--functions', help='name of the functions directory', required=True)
    parser.add_argument('--p', help='partition number', type=int, required=True )
    parser.add_argument('--labels', help='location of labels file', required=True)
    parser.add_argument('--output', help='output path', required=True)
    parser.add_argument('--gamma', help='gamma', default=.65, type=float)
    args = parser.parse_args()

    run = datetime.datetime.now().strftime("run-%Y-%m-%d-%R")
    ww = os.path.join(args.output, run)
    pathlib.Path(ww).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(filename=f"{ww}/info.log", level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    logging.info(args)

    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY',5*2147483648)
    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE', 5*134217728)
    # following can reduce the memory footprint
    tc.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', 1)

    logging.info(f"Loading functions from {args.functions}{args.p}")
    mw = tc.load_sframe(f"{args.functions}{args.p}")
    if 'fcount' in mw.column_names():
        mw.remove_column('fcount', inplace=True)

    if 'hapk' in mw.column_names():
        mw.rename(names={'hapk': 'apk'}, inplace=True)

    if 'hfunc' in mw.column_names():
        mw.rename(names={'hfunc': 'function'}, inplace=True)

    logging.info(f"Reading labels from {args.labels}")
    labels = pd.read_csv(args.labels, index_col=0)
    classifier = lambda x: int(labels.loc[x]['malware_label'])

    napks = mw['apk'].unique().to_numpy()
    logging.info(f"Got: {napks.shape[0]} apks")
    
    net = f_create_network(data=mw, gamma=args.gamma)
    #voting = convert_to_voting(net, classifier)
    save_nets({args.gamma: [net]}, f"{args.gamma}-{args.p}-tc-nets",  directory=ww)
    logging.info(f"Network with {len(net)} anchors saved ")

    anchors = get_anchor_coords(net=net, data=mw)
    pp = os.path.join(ww, f"anchors-{args.p}")
    anchors.save(pp, format='binary')
    logging.info(f"Anchor cords saved in {pp}")

