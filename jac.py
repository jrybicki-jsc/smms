from appknn import create_aggregating_net, jaccard, create_voting_net, lcl, adf
from typing import Sequence, Callable, NewType
from random import sample
from dataprep import join_tables, get_part_indexes
import pandas as pd
from appknn import evaluate_voting_net
from tqdm import tqdm


def setup_data():
    labels = pd.read_csv('data/labels_encoded.csv')

    v = pd.read_csv('data/functions_encoded.csv')
    funcs = v.groupby(by='apn')['nf'].apply(set)

    geef = join_tables(funcs=funcs, labels=labels)

    afs = geef['nf'].to_numpy()
    lbs = geef['mal'].to_numpy()
    return afs, lbs

def run_jaccard(train, test, distance, classifier):
    res = dict()
    nets = dict()
    
    for gamma in tqdm([0, 0.1, 0.2, 0.4, 0.5, 0.8, 0.85, 0.9, 1.0]):
        mv = create_voting_net(gamma=gamma, apns=train, distance=distance, classifier=classifier)
        false_negative, false_positives = evaluate_voting_net(apns=test, classifier=classifier, net=mv, distance=distance)
        res[gamma] = [false_negative, false_positives]
        nets[gamma] = mv.copy()

    sizes = {g:len(n.keys()) for g, n in nets.items()}
    perfs = pd.DataFrame.from_dict(res, orient='index', columns=['fp', 'fn'])
    szs = pd.DataFrame.from_dict(sizes, orient='index', columns=['sizes'])

    gj = perfs.join(szs)
    part_size = len(train)
    gj['compression'] = gj.sizes/part_size
    gj.to_csv(f"res/jac{part_size}.csv")

def run_euclid(train, test, distance,classifier):
    res = dict()
    nets = dict()
    distance = lambda x,y: adf(x, y, afs)

    for gamma in tqdm([0, 1, 2, 4, 8, 16, 32, 64, 128]):
        mv = create_voting_net(gamma=gamma, apns=train, distance=distance, classifier=classifier)
        false_negative, false_positives = evaluate_voting_net(apns=test, classifier=classifier, net=mv, distance=distance)
        res[gamma] = [false_negative, false_positives]
        nets[gamma] = mv.copy()
        
        sizes = {g:len(n.keys()) for g, n in nets.items()}
        perfs_di = pd.DataFrame.from_dict(res, orient='index', columns=['fp', 'fn'])
        szs_di = pd.DataFrame.from_dict(sizes, orient='index', columns=['sizes'])
        gj_euc = perfs_di.join(szs_di)
        part_size = len(train)
        gj_euc['compression'] = gj_euc.sizes/part_size
        gj_euc.to_csv(f"res/euc{part_size}.csv")

if __name__ == "__main__":
    print("Setting up data...")
    afs, lbs = setup_data()
    num_parts = 2
    part_size = 500
    parts = get_part_indexes(afs, num_parts, part_size)
    test_size = 100
    test_set = parts[1][:test_size]
    train_set = parts[0]
    
    print(f"Part size: {part_size} and {test_size=}")
    print(f"Jaccard experiments")
    distance = lambda x,y: jaccard(x, y, afs)
    classifier=lambda x: lcl(x, lbs)
    run_jaccard(train=train_set, test=test_set, distance=distance, classifier=classifier)

    print("Euclid experiments")
    distance = lambda x,y: adf(x, y, afs)
    run_euclid(train=train_set, test=test_set, distance=distance, classifier=classifier)

    
