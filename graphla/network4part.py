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


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Calculate network for a partitions')
    parser.add_argument('--functions', help='name of the input functions', required=True)
    parser.add_argument('--labels', help='location of labels file', required=True)
    parser.add_argument('--output', help='output path', required=True)
    parser.add_argument('--gamma', help='gamma', default=.65, type=float)
    args = parser.parse_args()
    print(args)

    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY',5*2147483648)
    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE', 5*134217728)
    # following can reduce the memory footprint
    tc.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', 1)

    print(f"Loading functions from {args.functions}")
    mw = tc.load_sframe(args.functions)
    if 'fcount' in mw.column_names():
        mw.remove_column('fcount', inplace=True)

    if 'hapk' in mw.column_names():
        mw.rename(names={'hapk': 'apk'}, inplace=True)

    if 'hfunc' in mw.column_names():
        mw.rename(names={'hfunc': 'function'}, inplace=True)

    print(f"Reading labels from {args.labels}")
    labels = pd.read_csv(args.labels, index_col=0)
    classifier = lambda x: int(labels.loc[x]['malware_label'])

    run = datetime.datetime.now().strftime("run-%Y-%m-%R")
    ww = os.path.join(args.output, run)
    pathlib.Path(ww).mkdir(parents=True, exist_ok=True)
    
    napks = mw['apk'].unique().to_numpy()
    print(f"Got: {napks.shape[0]} apks")
    
    net = f_create_network(data=mw, gamma=args.gamma)
    voting = convert_to_voting(net, classifier)

    save_nets({args.gamma: [net, voting]}, f"{gamma}-tc-nets", , directory=ww)
