# spliter
import numpy as np
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import turicreate as tc
import argparse
import pathlib
import datetime
import os
from grapm import partition_ndframe
import datetime


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Split data into partitions')
    parser.add_argument('--functions', help='name of the input functions', required=True)
    parser.add_argument('--p', help='number of partitions', default=4, type=int)
    parser.add_argument('--output', help='output path', required=True)
    args = parser.parse_args()
    print(args)

    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY',5*2147483648)
    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE', 5*134217728)
    # following can reduce the memory footprint
    tc.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', 8)

    print(f"Loading functions from {args.functions}")
    mw = tc.load_sframe(args.functions)
    if 'fcount' in mw.column_names():
        mw.remove_column('fcount', inplace=True)

    if 'hapk' in mw.column_names():
        mw.rename(names={'hapk': 'apk'}, inplace=True)

    if 'hfunc' in mw.column_names():
        mw.rename(names={'hfunc': 'function'}, inplace=True)

    #subsamp = get_sample(mw=mw, frac=0.2)
    
    napks = mw['apk'].unique().to_numpy()
    print(f"Got: {napks.shape[0]} apks")
    test_size = 1000
    train, test = train_test_split(napks, test_size=test_size, random_state=42)
    np.save(f"{args.output}/test-tc-{test_size}", test)

    print(f"Spliting into {args.p} partitions")
    parts = partition_ndframe(nd=train, n_parts=args.p)
    for i, part in enumerate(parts):
        output = os.path.join(args.output, f"part-{i}")
        print(f"Saving to {output}")
        mw.filter_by(values=part, column_name='apk').save(output, format='binary')
    
    

    
    