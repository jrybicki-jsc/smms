# spliter
import turicreate as tc
import argparse
import os
from grapm import f_create_network, convert_to_voting, save_nets
import logging
from streamed import get_anchor_coords
from utils import setup_logging, setup_path, load_functions_partition, setup_turi


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Calculate network for a partitions')
    parser.add_argument('--functions', help='name of the functions directory', required=True)
    parser.add_argument('--p', help='partition number', type=int)
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


    mw = load_functions_partition(directory=args.functions, name=args.p)
    logging.info(f"Stargng network calculation for gamma={gamma}")
        
    net = f_create_network(data=mw, gamma=gamma)
    save_nets({args.gamma: [net]}, f"{args.gamma}-{args.p}-tc-nets",  directory=path)
    logging.info(f"Network with {len(net)} anchors saved ")

    anchors = get_anchor_coords(net=net, data=mw)
    pp = os.path.join(path, f"anchors-{args.p}")
    anchors.save(pp, format='binary')
    logging.info(f"Anchor cords saved in {pp}")

