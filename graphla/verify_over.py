import pandas as pd
import numpy as np
#import graphlab as tc
import turicreate as tc
from tqdm.notebook import tqdm
import turicreate.aggregate as agg
import pickle

if __name__=="__main__":
    test_apns = np.load('../res/test-tc-1000.npy')

    net_file = '../res/9003-tc-jaccard-votingnets.pickle'
    with open(net_file, 'rb') as f:
        nets = pickle.load(f)
    
    gamma = 0.0
    ref, mer = nets[0.0]
    overlap_ref = [t for t in test_apns if t in ref]
    overlap_mer = [t for t in test_apns if t in mer]
    
    print(f"Overlap ref: {len(overlap_ref)}")
    print(f"Overlap mer: {len(overlap_mer)}")
    #print(f"Ref. overlap: {len(overlap_ref)}")


    