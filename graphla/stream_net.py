#stream nets
import argparse
import datetime
import logging
import os
import pathlib
import pickle

import turicreate as tc
import turicreate.aggregate as agg
from grapm import save_nets
from utils import (load_functions_partition, setup_logging, setup_path,
                   setup_turi, load_net)


def tc_based_nn(net, anchors, partition):
    #allt = data.filter_by(values=anch, column_name='apk')
    allt = anchors.append(partition)

    #m = allt.shape[0] # overdoing it here a little?
    m = 20 * len(net)
    logging.info(f"Setting topk to {m}")
    
    sim_recom = tc.item_similarity_recommender.create(
        allt, 
        user_id='function', 
        item_id='apk', 
        similarity_type='jaccard', 
        degree_approximation_threshold=15*4096,
        only_top_k=m, verbose=False)
    
    # smaller k could be an optimization here
    apks = partition['apk'].unique()
    items =sim_recom.get_similar_items(apks, k=m)
    # recomendations excluding network anchors (apk, exclude)
    fitems = items.filter_by(values=list(net.keys()), column_name='similar')
    
    return fitems.groupby(key_column_names=['apk'], 
                          operations={'nn': tc.aggregate.ARGMAX('score', 'similar')})


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Stream networks')
    parser.add_argument('--net', help='orgin net', required=True)
    parser.add_argument('--anchors', help='orgin anchors', required=True)
    parser.add_argument('--functions', help='name of the input functions partition', required=True)
    parser.add_argument('--p', help='partition number', type=int, required=True)
    parser.add_argument('--output', help='output path', required=True)
    args = parser.parse_args()

    path = setup_path(args=args)
    setup_logging(path=path, parser=parser)
    setup_turi()

    logging.info(f"Loading origin network {args.net} & {args.anchors}")
    gamma, net = load_net(args.net)

    logging.info('Loading anchors')
    an = tc.load_sframe(args.anchors)

    logging.info('Loading paritiion')
    mw = load_functions_partition(directory=args.functions, name=args.p)

    logging.info('Nearest neigbour search')
    neigh = tc_based_nn(net=net, anchors=an, partition= mw)
    
    logging.info('Conversion')
    dicted = neigh.groupby(key_column_names='nn', operations={'nodes': agg.DISTINCT('apk')})
    true_dicts = {row['nn']: row['nodes'] for row in dicted}

    logging.info('Saving')
    save_nets({gamma: [true_dicts]}, f"{gamma}-streamed-{args.p}",  directory=path)
    logging.info(f"Saved network with {len(true_dicts)}")
    