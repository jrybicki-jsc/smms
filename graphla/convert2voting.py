import argparse
import logging
import pickle
import pandas as pd

import turicreate as tc

from utils import setup_path, setup_logging, setup_turi
from grapm import convert_to_voting, save_nets

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Convert to voting network')
    parser.add_argument('--net', help='networks directory', required=True)
    parser.add_argument('--labels', help='apk labels', required=True)
    parser.add_argument('--output', help='output path', required=True)
    args = parser.parse_args()

    path = setup_path(args)
    setup_logging(path=path, parser=parser)

    setup_turi()

    logging.info(f"Reading labels from {args.labels}")
    labels = pd.read_csv(args.labels, index_col=0)
    classifier = lambda x: int(labels.loc[x]['malware_label'])

    #0.86-0-tc-nets.pickle
    with open(f"{args.net}", 'rb') as f:
        net = pickle.load(f)
    
    gamma = list(net.keys())[0]
    net = list(net.values())[0][0]
    voting = convert_to_voting(net, classifier)
    # args.nets will include path so we trick it with directory =''
    voting_path = args.net.replace('.pickle', '-voting')
    save_nets({gamma: [voting]}, voting_path,  directory='')
    logging.info(f"Voting network with {len(voting)} anchors saved ")