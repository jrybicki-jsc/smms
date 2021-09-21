
import argparse
import logging
import os

from grapm import f_create_network, save_nets
from streamed import get_anchor_coords
from utils import (load_functions_partition, setup_logging, setup_path,
                   setup_turi)

from net_strip import strip_net

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Lited Calculate network for a partitions')
    parser.add_argument('--functions', help='name of the functions directory', required=True)
    parser.add_argument('--p', help='partition number', type=int)
    parser.add_argument('--output', help='output path', required=True)
    parser.add_argument('--list', nargs='+', help='Gamma list', required=True)
    args = parser.parse_args()

    path = setup_path(args=args)
    setup_logging(path=path, parser=parser)

    setup_turi()
    mw = load_functions_partition(directory=args.functions, name=args.p)

    for gamma in args.list:
        t_gamma = gamma
        if gamma > 10:
            gamma = gamma/10.0

        if gamma > 1.0:
            gamma = gamma/10.0

        logging.info("Stargng network calculation for gamma=%s (%s)", gamma, t_gamma)
        net = f_create_network(data=mw, gamma=gamma)
        save_nets({t_gamma: [net]}, f"{t_gamma}-{args.p}-tc-nets",  directory=path)
        logging.info("Network with %d anchors saved ", len(net))

        anchors = get_anchor_coords(net=net, data=mw)
        pp = os.path.join(path, f"anchors-{args.p}")
        anchors.save(pp, format='binary')
        logging.info("Anchor cords saved in %s", pp)

        st = strip_net(net)
        if args.output:
            save_nets({t_gamma: [st]}, "stripped",  directory=args.output)
        else:
            fname = args.net.replace('.pickle', '-stripped')
            save_nets({t_gamma: [st]}, fname,  directory='')
