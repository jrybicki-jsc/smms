import logging
import turicreate as tc
import os
import datetime
import pathlib

def load_functions_partition(args):
    logging.info(f"Loading functions from {args.functions}{args.p}")
    mw =  tc.load_sframe(f"{args.functions}{args.p}")
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

def setup_logging(path, args):
    logging.basicConfig(filename=f"{path}/info.log", filemode='a', level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    logging.info(args)
    