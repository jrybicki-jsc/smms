import pickle
from appknn import classify_using_voting, vote, adf, lcl, calculate_metrics, eval_net, jaccard
from dataprep import precision, recall, r1
import pandas as pd
from tqdm import tqdm
import argparse

def get_data():
    m = pd.read_csv('data/joined.csv', index_col='apn')
    m['nf'] = m['nf'].apply(eval)
    return m

def add_mets(mdf):
    mdf['precision'] = mdf.apply(precision, axis=1)
    mdf['recall'] = mdf.apply(recall, axis=1)
    mdf['r1'] = mdf.apply(r1, axis=1)

def get_distance(net_filename: str):
    if 'jaccard' in net_filename:
        return jaccard
    else:
        return adf

def get_sizes(nets):
    sizes = dict()
    for gamma, [mer, ref] in nets.items():
        sizes[gamma] = [len(mer.keys()), len(ref.keys())]

    return pd.DataFrame.from_dict(sizes, orient='index', columns=['anchors_mer', 'anchors_ref'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process networks.')
    parser.add_argument('--net-file', help='name of the network file', required=True)
    parser.add_argument('--test-file', help='name of the test file', required=True)
    parser.add_argument('--out', help='name of the output', default='res/out.csv')
    args = parser.parse_args()
    print(args)
    distance_function = get_distance(args.net_file)

    with open(args.net_file, 'rb') as f:
        nets = pickle.load(f)

    test = pd.read_csv(args.test_file, index_col=0)
    m = get_data()
    classifier = lambda x: lcl(x, m['ml'])
    distance = lambda x,y: distance_function(x,y, m['nf'])

    merges = dict()
    refs = dict()
    for gamma, [me, ref] in tqdm(nets.items()):
        refs[gamma] = eval_net(net=ref, test_set=test.index, distance=distance, classifier=classifier)
        merges[gamma] = eval_net(net=me, test_set=test.index, distance=distance, classifier=classifier)

    mdf = pd.DataFrame.from_dict(merges, orient='index', columns=['tp', 'fp', 'tn', 'fn'])
    mdr = pd.DataFrame.from_dict(refs, orient='index', columns=['tp', 'fp', 'tn', 'fn'])
    add_mets(mdf)
    add_mets(mdr)

    sizes = get_sizes(nets)
    alldata = mdf.join(mdr, lsuffix='_mer', rsuffix='_ref').join(sizes)

    alldata.to_csv(args.out)

