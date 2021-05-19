import numpy as np
import turicreate as tc
import pickle
import argparse
from utils import (load_functions_partition, setup_logging, setup_path,
                   setup_turi)
import logging

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Extract functions for given apks e.g. from a merged network')
    parser.add_argument('--functions', help='name of the function file', required=True, default='/storage/users/cnalab/apkdata-tanya/binary/new.large.sframe')
    parser.add_argument('--net', help='name of a network file (extraction only for anchors)')
    parser.add_argument('--output', help='output path', required=True)
    args = parser.parse_args()
    #test_file = '/storage/users/cnalab/apkdata-tanya/binary/test-tc-1000.npy'
    
    #if test_file:
    #    print(f"Reading test file: {test_file}")
    #    test_apns = np.load(test_file)

    path = setup_path(args=args)
    setup_logging(path=path, parser=parser)
    net_file = args.net 
    logging.info(f"Reading reading net file {net_file}")
    with open(net_file, 'rb') as f:
        net = pickle.load(f)
    net = list(net.values())[0][0]
    test_apns = list(net.keys())

    logging.info(f"Extracted apn: {len(test_apns)}")

    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY',15*2147483648)
    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE', 15*134217728)
    # following can reduce the memory footprint
    tc.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', 16)

    logging.info('Loading functions')
    mw = load_functions_partition(directory='', name=args.functions)

    logging.info('Filter started')
    test_f = mw.filter_by(values=test_apns, column_name='apk')
    test_f.save(f"{path}/convert-out-{len(test_apns)}", format='binary')