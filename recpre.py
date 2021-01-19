# prec/rec
import pandas as pd
import numpy as np
import pickle
from appknn import classify_using_voting, vote, adf, lcl, calculate_metrics, eval_net, jaccard
from tqdm import tqdm

def get_data():
    m = pd.read_csv('data/joined.csv', index_col='apn')
    m['nf'] = m['nf'].apply(eval)
    return m

def get_votes(net, apns, distance):
    return [classify_using_voting(app=a, net=net, distance=distance, k=1) for a in apns]


def conver_to_probs(votes):
    return [1.0 - v[1]/(v[0]+v[1]) for v in votes]

def get_predictions_score(net, distance, test):
    votes = get_votes(net=net, apns=test, distance=distance)
    return conver_to_probs(votes)

if __name__=="__main__":
    #net_file = 'res/9503-jaccard-votingnets.pickle'
    #test_file ='res/10003-test.csv'
    net_file = 'res/9003-tc-jaccard-votingnets.pickle'
    test_file = 'res/test-tc-1000.npy'

    #test = pd.read_csv(test_file, index_col=0)
    test = np.load(test_file)
    with open(net_file, 'rb') as f:
        nets = pickle.load(f)
        
    m = get_data()

    distance = lambda x,y: jaccard(x,y, m['nf'])
    classifier = lambda x: lcl(x, m['ml'])

    refs = dict()
    mers = dict()
    #test_a = list(test.index)
    test_a  = list(test)
    for gamma, [mer, ref] in tqdm(nets.items()):
        mers[gamma] = get_predictions_score(net=mer, distance=distance, test=test_a)
        refs[gamma] = get_predictions_score(net=ref, distance=distance, test=test_a)
        with open(f"/tmp/preds-{gamma}-mers.pickle", 'wb+') as f:
            pickle.dump(mers[gamma], f)
    
        with open(f"/tmp/preds-{gamma}-refs.pickle", 'wb+') as f:
            pickle.dump(refs[gamma], f)
    
    with open(f"res/tc-preds-mers.pickle", 'wb+') as f:
        pickle.dump(mers, f)
    
    with open(f"res/tc-preds-refs.pickle", 'wb+') as f:
        pickle.dump(refs, f)
        
    

