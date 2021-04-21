#!/usr/bin/env python
import pandas as pd
import numpy as np
#import graphlab as tc
import turicreate as tc
import argparse



if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Convert apk file.')
    parser.add_argument('--input', help='name of the input file', required=True)
    parser.add_argument('--output', help='name of the output', required=True)
    #fname = '../data/sample_10000_vt_mal_2017_2020_az_2020_benign_hashed_md5.csv'
    args = parser.parse_args()
    print(args)

    mw = tc.SFrame.read_csv(args.input, header=False, verbose=False)
    print(mw.head())

    mw['hapk'] = mw['X1'].apply(lambda x: hash(str.upper(x)))
    mw['hfunc'] = mw['X2'].apply(lambda x: hash(str.upper(x)))
    clk = mw.remove_columns(['X1', 'X2', 'X3'])
    print(clk.head())

    clk.save(args.output, format='binary')