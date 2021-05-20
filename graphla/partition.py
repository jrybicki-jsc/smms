# spliter
import numpy as np
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
import logging
from utils import setup_logging, setup_path, setup_turi, load_functions_partition


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Split data into partitions')
    parser.add_argument('--functions', help='name of the input functions', required=True)
    parser.add_argument('--p', help='number of partitions', default=4, type=int)
    parser.add_argument('--test', help='size of test part', default=1000, type=int)
    parser.add_argument('--output', help='output path', required=True)
    args = parser.parse_args()
    path = setup_path(args)
    setup_logging(path=path, parser=parser)
    setup_turi()

    logging.info(f"Loading functions from {args.functions}")
    mw = load_functions_partition(directory='', name=args.functions)
    #subsamp = get_sample(mw=mw, frac=0.2)
    
    napks = mw['apk'].unique().to_numpy()
    print(f"Got: {napks.shape[0]} apks")
    test_size = args.test 
    if test_size>0:
        logging.info("Creating test file ")
        train, test = train_test_split(napks, test_size=test_size, random_state=42)
        np.save(f"{args.output}/test-tc-{test_size}", test)

    print(f"Spliting into {args.p} partitions")
    parts = partition_ndframe(nd=train, n_parts=args.p)
    for i, part in enumerate(parts):
        output = os.path.join(args.output, f"part-{i}")
        print(f"Saving to {output}")
        mw.filter_by(values=part, column_name='apk').save(output, format='binary')
    