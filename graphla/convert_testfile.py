import numpy as np 
import turicreate as tc 

if __name__=="__main__":
    test_file = ''
    functions = ''

    print(f"Reading test file: {test_file}")
    test_apns = np.load(test_file)
    
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