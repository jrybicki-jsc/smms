# spliter
import argparse
import logging
import os

import numpy as np

from grapm import save_nets
from streamed import get_anchor_coords
from utils import (load_functions_partition, setup_logging, setup_path,
                   setup_turi)


def random_net(apks, size):
    
    return {a:[] for a in np.random.choice(a=apks, size=size, replace=False)}



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Calculate random set from a partition')
    parser.add_argument('--functions', help='name of the functions directory', required=True)
    parser.add_argument('--output', help='output path', required=True)
    #parser.add_argument('--size', help='size', default=120, type=int)
    parser.add_argument('--list', nargs='+', help='Sizes list', required=True)
    args = parser.parse_args()
   
    path = setup_path(args=args)
    setup_logging(path=path, parser=parser)
    setup_turi()

    mw = load_functions_partition(directory=args.functions, name='')
    apks = mw['apk'].unique()

    for size in args.list:
        size = int(size)
        logging.info(f"Stargng network creation for size={size}")
        net = random_net(apks=apks, size=size)
        save_nets({size: [net]}, f"{size}-random-nets",  directory=path)
        logging.info(f"Network with {len(net)} anchors saved ")

        anchors = get_anchor_coords(net=net, data=mw)
        pp = os.path.join(path, f"anchors-{size}")
        anchors.save(pp, format='binary')
        logging.info(f"Anchor cords saved in {pp}")
