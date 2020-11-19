# feature importance
import pandas as pd
import numpy as np
import pickle
from appknn import classify_using_voting, vote, adf, lcl, calculate_metrics, jaccard, eval_net, create_voting_net
from evaluate import get_data
from dataprep import get_part_indexes
from tqdm import tqdm


def mjaccard(x,y, nafs):
    if len(nafs[x]) == 0 == len(nafs[y]):
        return 1
    return jaccard(x,y, nafs)

def enable_feature(di, feature, fullset):
    ret = di.copy()
    for apn, functions in ret.items():
        if feature in fullset[apn]:
            functions.add(feature)
    return ret

if __name__ == "__main__":
    d = get_data()
    nafs = d['nf'].to_numpy()
    nlabs = d['ml'].to_numpy()
    nc = lambda x: lcl(x, d['ml'])

    sample_size = 800
    test_size = 50
    a= get_part_indexes(d['nf'], num_parts=1, size=sample_size, seed=42)[0]

    smallset = d['nf'].iloc[a]

    allfeatures = list(set(smallset.values[0]))
    bestfeatures = list()
    star = {apn: set() for apn in smallset.index}
    removals = list()
    for i in tqdm(range(100)):
        test_set = list(smallset.index)[-test_size:]
        res = dict()
        for feature in allfeatures:
            a = enable_feature(star, feature, smallset)
            train_star = {apn: fea for apn, fea in a.items() if apn in list(smallset.index)[:- test_size]}
            vn = create_voting_net(gamma=0.7, apns=list(train_star.keys()), 
                                distance=lambda x,y: mjaccard(x,y, train_star), classifier=nc)

            #TP, FP, TN, FN
            e0, e1, e3, e2 = eval_net(net=vn, test_set=test_set, 
                                    distance=lambda x,y: mjaccard(x,y, star), classifier=nc)

            res[feature] = e1 + e2

        bestfeat = min(res, key=res.get)
        
        star = enable_feature(star, bestfeat, smallset)
        print(f"Best performing feature is {bestfeat} {res[bestfeat]} (worst {max(res, key=res.get)})")
        # remove from features
        allfeatures.remove(bestfeat)
        removals.append(res)
        bestfeatures.append(bestfeat)
    print(bestfeatures)
    with open(f"res/bestfeatures-{sample_size}.pickle", 'wb+') as f:
            pickle.dump(bestfeatures, f)



