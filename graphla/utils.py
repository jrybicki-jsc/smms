import logging
import turicreate as tc
import os
import datetime
import pathlib
import pickle

def load_functions_partition(directory, name):
    if name is None:
        name = ''
    logging.info(f"Loading functions from {directory}{name}")
    mw =  tc.load_sframe(f"{directory}{name}")
    if 'fcount' in mw.column_names():
        mw.remove_column('fcount', inplace=True)

    if 'hapk' in mw.column_names():
        mw.rename(names={'hapk': 'apk'}, inplace=True)

    if 'hfunc' in mw.column_names():
        mw.rename(names={'hfunc': 'function'}, inplace=True)

    return mw

def load_net(path):
    logging.info(f"Loading network from {path}")
    with open(path, 'rb') as f:
        net = pickle.load(f)
    gamma = list(net.keys())[0]
    net = list(net.values())[0][0]
    return gamma, net

def setup_path(args, time: bool=True):
    if time:
        run = datetime.datetime.now().strftime("run-%Y-%m-%d-%R")
    else:
        run =''
    
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
