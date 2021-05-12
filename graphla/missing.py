import argparse
import logging
import pandas as pd
import turicreate as tc
from utils import setup_path, setup_logging

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Find missing labels')
    parser.add_argument('--functions', help='functions', required=True)
    parser.add_argument('--labels', help='apk labels', required=True)
    parser.add_argument('--output', help='output path', required=True)
    args = parser.parse_args()

    path = setup_path(args)
    setup_logging(path=path, args=args)

    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY',15*2147483648)
    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE', 15*134217728)
    # following can reduce the memory footprint
    tc.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)

    logging.info(f"Reading labels from {args.labels}")
    labels = pd.read_csv(args.labels, index_col=0)
    
    logging.info(f"Loading functions from {args.functions}")
    mw2 = tc.load_sframe(args.functions)
    apks = mw2['apk'].unique()

    apkout = f"{args.output}/apks/"
    logging.info(f"Saving apks to {apkout}")
    apks.save(apkout)

    missing = apks.filter_by(values=list(labels.index), exclude=True)
    missout = f"{args.output}/missing"
    logging.info(f"Found {missing.shape[0]} missing labels. Saving to {missout}")
    missing.save(missout)
