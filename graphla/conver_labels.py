#!/usr/bin/env python
import pandas as pd
import argparse
import hashlib

def calc_hash(x, digits=12):
    return int(hashlib.sha1(x.upper().encode("utf-8")).hexdigest(), 16) % (10 ** digits)

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Convert labels file.')
    parser.add_argument('--input', help='name of the input labels', required=True)
    parser.add_argument('--output', help='output path', required=True)
    args = parser.parse_args()
    print(args)

    labels = pd.read_csv(args.input)
    print(labels.head())

    labels['hapk']=labels['apk'].apply(calc_hash)
    labels.drop(columns=['apk'], inplace=True)
    r = labels.rename(columns={'hapk': 'apk'}).set_index('apk')

    with open(args.output, 'w+') as f:
        r.to_csv(f)
    
    
    #alternative: using pandas
    #tp = pd.read_csv('large_dataset.csv', iterator=True, chunksize=1000)  # gives TextFileReader, which is iterable with chunks of 1000 rows.
    #df = pd.concat(tp, ignore_index=True)