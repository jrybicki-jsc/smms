import argparse
import logging
import pickle
import pandas as pd

import turicreate as tc

from utils import setup_path, setup_logging, load_functions_partition
from grapm import convert_to_voting, save_nets

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Convert to voting network')
    parser.add_argument('--nets', help='networks directory', required=True)
    parser.add_argument('--p', help='partition number', type=int, required=True)
    parser.add_argument('--labels', help='apk labels', required=True)
    parser.add_argument('--output', help='output path', required=True)
    args = parser.parse_args()

    path = setup_path(args)
    setup_logging(path=path, args=args)

    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY',5*2147483648)
    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE', 5*134217728)
    # following can reduce the memory footprint
    tc.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)

    logging.info(f"Reading labels from {args.labels}")
    labels = pd.read_csv(args.labels, index_col=0)
    classifier = lambda x: int(labels.loc[x]['malware_label'])

    #0.86-0-tc-nets.pickle
    with open(f"{args.nets}{args.p}-tc-nets.pickle", 'rb') as f:
        net = pickle.load(f)

    net = list(net.values())[0][0]
    voting = convert_to_voting(net, classifier)
    # args.nets will include path so we trick it with directory =''
    save_nets({0.0: [voting]}, f"{args.nets}-{args.p}-voting-net",  directory='')
    logging.info(f"Voting network with {len(voting)} anchors saved ")
    
