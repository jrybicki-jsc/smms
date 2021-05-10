import argparse
import logging

import turicreate as tc

from utils import setup_path, setup_logging, load_functions_partition

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Merge partitions')
    parser.add_argument('--functions', help='name of the functions directory', required=True)
    parser.add_argument('--p1', help='partition number', type=int, required=True)
    parser.add_argument('--p2', help='partition number', type=int, required=True)
    parser.add_argument('--output', help='output path', required=True)
    args = parser.parse_args()

    path = setup_path(args)
    setup_logging(path=path, args=args)

    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY',5*2147483648)
    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE', 5*134217728)
    # following can reduce the memory footprint
    tc.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)

    mw1 = load_functions_partition(directory=args.functions, name=args.p1)
    logging.info(f"Read {mw1.num_rows()} rows")

    mw2 = load_functions_partition(directory=args.functions, name=args.p2)
    logging.info(f"Read {mw2.num_rows()} rows")

    mw = mw1.append(mw2)
    
    mw.save(f"{path}/merged-{args.p1}-{args.p2}.csv", format='binary')
    logging.info(f"saved {mw.num_rows()} rows")