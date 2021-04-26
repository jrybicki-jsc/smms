#!/usr/bin/env python
import pandas as pd
import numpy as np
#import graphlab as tc
import turicreate as tc
import argparse
import glob 
import os
from grapm import f_create_network
import pickle


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Convert apk file.')
    parser.add_argument('--input', help='name of the input path', required=True)
    args = parser.parse_args()
    print(args)

    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY',5*2147483648)
    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE', 5*134217728)
    # following can reduce the memory footprint
    tc.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', 24)

    print(f"Reading from {args.input}")
    mw = tc.load_sframe(args.input)
    #mw.remove_column('fcount', inplace=True)
    mw.rename(names={'hapk': 'apk', 'hfunc': 'function'}, inplace=True)
    print("Starging network creation")
    net = f_create_network(data=mw, gamma=0.65)

    print('saving network')
    with open(f"../res/very_large_net.pickle", 'wb+') as f:
        pickle.dump(net, f)
    