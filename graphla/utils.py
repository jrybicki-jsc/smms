import logging
import turicreate as tc
import os
import datetime
import pathlib

def load_functions_partition(directory, name):
    logging.info(f"Loading functions from {directory}{name}")
    mw =  tc.load_sframe(f"{directory}{name}")
    if 'fcount' in mw.column_names():
        mw.remove_column('fcount', inplace=True)

    if 'hapk' in mw.column_names():
        mw.rename(names={'hapk': 'apk'}, inplace=True)

    if 'hfunc' in mw.column_names():
        mw.rename(names={'hfunc': 'function'}, inplace=True)

    return mw


def setup_path(args):
    run = datetime.datetime.now().strftime("run-%Y-%m-%d-%R")
    ww = os.path.join(args.output, run)
    pathlib.Path(ww).mkdir(parents=True, exist_ok=True)
    return ww

def setup_logging(path, parser):
    logging.basicConfig(filename=f"{path}/info.log", filemode='a', level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    logging.info(parser.description)
    logging.info(parser.parse_args())
    
def setup_turi():
    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY',5*2147483648)
    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE', 5*134217728)
    # following can reduce the memory footprint
    tc.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', 4)
