import numpy as np 
import turicreate as tc
import pickle

if __name__=="__main__":
    test_file = ''
    functions = ''
    net_file =''

    if test_file:
        print(f"Reading test file: {test_file}")
        test_apns = np.load(test_file)
    else:
        print(f"Reading reading net file {net_file}")
        with open(net_file, 'rb') as f:
            net = pickle.load(f)
        net = list(net.values())[0][0]
        test_apns = list(net.keys())

    print(f"Test apn: {test_apns}")

    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY',15*2147483648)
    tc.config.set_runtime_config('TURI_FILEIO_MAXIMUM_CACHE_CAPACITY_PER_FILE', 15*134217728)
    # following can reduce the memory footprint
    tc.config.set_runtime_config('TURI_DEFAULT_NUM_PYLAMBDA_WORKERS', 16)

    print(f"Loading functions from {functions}")
    mw = tc.load_sframe(functions)
    if 'fcount' in mw.column_names():
        mw.remove_column('fcount', inplace=True)

    if 'hapk' in mw.column_names():
        mw.rename(names={'hapk': 'apk'}, inplace=True)

    if 'hfunc' in mw.column_names():
        mw.rename(names={'hfunc': 'function'}, inplace=True)

    test_f = mw.filter_by(values=test_apns, column_name='apk')
    test_f.save('bin-test', format='binary')