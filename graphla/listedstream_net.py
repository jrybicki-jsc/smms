#listed stream nets
import argparse
import logging
import os

import turicreate as tc
import turicreate.aggregate as agg
from grapm import save_nets
from stream_net import tc_based_nn
from utils import (load_functions_partition, load_net, setup_logging,
                   setup_path, setup_turi)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Listed stream networks')
    parser.add_argument('--dir', help='orgin net & anchors', required=True)
    #parser.add_argument('--anchors', help='orgin anchors', required=True)
    parser.add_argument('--functions', help='name of the input functions partition', required=True)
    parser.add_argument('--p', help='partition number', type=int, required=True)
    parser.add_argument('--output', help='output path', required=True)
    parser.add_argument('--list', nargs='+', help='Gamma list', required=True)

    args = parser.parse_args()

    path = setup_path(args=args, time=False)
    setup_logging(path=path, parser=parser)
    setup_turi()
    logging.info('Loading paritiion')
    mw = load_functions_partition(directory=args.functions, name=args.p)


    for gamma in args.list:
        t_gamma = gamma
        gamma = float(gamma)
        if gamma > 10:
            gamma = gamma/10.0

        if gamma > 1.0:
            gamma = gamma/10.0

        pp = os.path.join(args.dir, f"anchors-{args.p}")
        nn = os.path.join(args.dir, f"{t_gamma}-{args.p}-tc-nets")

        logging.info(f"Loading origin network from {args.dir} {nn} & {pp}")
        _, net = load_net(nn)
        logging.info('Loading anchors')
        an = tc.load_sframe(pp)

        logging.info('Nearest neigbour search')
        neigh = tc_based_nn(net=net, anchors=an, partition=mw)

        logging.info('Conversion')
        dicted = neigh.groupby(key_column_names='nn', operations={'nodes': agg.DISTINCT('apk')})
        true_dicts = {row['nn']: row['nodes'] for row in dicted}

        logging.info('Saving')
        save_nets({gamma: [true_dicts]}, f"{t_gamma}-streamed-{args.p}",  directory=path)
        logging.info(f"Saved network with {len(true_dicts)}")
