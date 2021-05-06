# spliter
import argparse
import logging

import turicreate as tc
from turicreate import aggregate as agg

from utils import setup_path, setup_logging, load_functions_partition

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Calculate network for a partitions')
    parser.add_argument('--functions', help='name of the functions directory', required=True)
    parser.add_argument('--p', help='partition number', type=int, required=True )
    parser.add_argument('--output', help='output path', required=True)
    args = parser.parse_args()

    path = setup_path(args)

    setup_logging(path=path, args=args)


    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY',5*2147483648)
    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE', 5*134217728)
    # following can reduce the memory footprint
    tc.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)

    mw = load_functions_partition(args=args)
    logging.info(f"Read {mw.num_rows} rows")

    ags = mw.groupby(key_column_names='apk', operations={'fcount': agg.COUNT()})
    ags.save(f"{path}/apks.csv", format='csv')

    fgs = mw.groupby(key_column_names='function', operations={'acount': agg.COUNT()})
    fgs.save(f"{path}/funcs.csv", format='csv')






    
    

    

