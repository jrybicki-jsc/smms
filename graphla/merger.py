#!/usr/bin/env python
import pandas as pd
import numpy as np
#import graphlab as tc
import turicreate as tc
import argparse
import glob 
import os

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Binary merger')
    parser.add_argument('--input', help='name of the input path', required=True)
    parser.add_argument('--output', help='output name', required=True)
    #fname = '../data/sample_10000_vt_mal_2017_2020_az_2020_benign_hashed_md5.csv'
    args = parser.parse_args()
    print(args)

    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY',5*2147483648)
    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE', 5*134217728)
    # following can reduce the memory footprint
    tc.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', 24)

    fls = list(glob.glob(f"{args.input}*"))
    print(fls)
    out = tc.load_sframe(fls.pop())

    for el in fls:
        print(el, out.shape)
        tm = tc.load_sframe(el)
        out = out.append(tm)

    out.save(args.output, format='binary')