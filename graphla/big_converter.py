#!/usr/bin/env python
import pandas as pd
import numpy as np
#import graphlab as tc
import turicreate as tc
import argparse
import glob 
import os

def convert_csv(fname, output):
    mw = tc.SFrame.read_csv(fname, header=False, verbose=True)#, nrows=100)
    print(mw.head())

    mw['hapk'] = mw['X1'].apply(lambda x: hash(str.upper(x)))
    mw['hfunc'] = mw['X2'].apply(lambda x: hash(str.upper(x)))
    clk = mw.remove_columns(['X1', 'X2', 'X3'])
    print(clk.head())

    clk.save(output, format='binary')



if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Convert apk file.')
    parser.add_argument('--input', help='name of the input path', required=True)
    parser.add_argument('--output', help='output path', required=True)
    #fname = '../data/sample_10000_vt_mal_2017_2020_az_2020_benign_hashed_md5.csv'
    args = parser.parse_args()
    print(args)

    for f in glob.glob(args.input+'*'):
        outnn = f.split('/')[-1]
        outnn = os.path.join(args.output, outnn)
        print(f"Processing {f} -> {outnn}")
        
        convert_csv(fname=f, output=outnn)


    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY',5*2147483648)
    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE', 5*134217728)
    # following can reduce the memory footprint
    #tc.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', 24)
    
    #alternative: using pandas
    #tp = pd.read_csv('large_dataset.csv', iterator=True, chunksize=1000)  # gives TextFileReader, which is iterable with chunks of 1000 rows.
    #df = pd.concat(tp, ignore_index=True)