# spliter
import numpy as np
from sklearn.model_selection import train_test_split
import turicreate as tc
import argparse
import os
from grapm import partition_ndframe
import logging
from utils import setup_logging, setup_path, setup_turi, load_functions_partition


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Split data into partitions')
    parser.add_argument('--functions', help='name of the input functions', required=True)
    parser.add_argument('--p', help='number of partitions', default=4, type=int)
    parser.add_argument('--test', help='size (or proportion) of test part', default=1000, type=float)
    parser.add_argument('--output', help='output path', required=True)
    parser.add_argument('--seed', help='random seed', default=42, type=int)
    args = parser.parse_args()
    path = setup_path(args)
    setup_logging(path=path, parser=parser)
    setup_turi()

    mw = load_functions_partition(directory='', name=args.functions)
    
    napks = mw['apk'].unique().to_numpy()
    logging.info(f"Got: {napks.shape[0]} apks")
    test_size = args.test 
    if test_size>0:
        logging.info("Creating test file")
        train, test = train_test_split(napks, test_size=test_size, random_state=args.seed)
        #np.save(f"{args.output}/test-tc-{test_size}", test)
        mw.filter_by(values=test, column_name='apk').save(f"{args.output}/test-tc-{test_size}", format='binary')
    else:
        train = napks


    logging.info(f"Spliting into {args.p} partitions")
    parts = partition_ndframe(nd=train, n_parts=args.p, seed=args.seed)
    for i, part in enumerate(parts):
        output = os.path.join(args.output, f"part-{i}")
        logging.info(f"Saving to {output}")
        mw.filter_by(values=part, column_name='apk').save(output, format='binary')
    