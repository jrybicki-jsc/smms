import pandas as pd
import numpy as np
#import graphlab as gl
import turicreate as tc
import turicreate.aggregate as agg
import time
from tqdm import tqdm
from collections import defaultdict
from numpy.random import default_rng
from sklearn.model_selection import train_test_split
from functools import partial

from multiprocessing import Pool
import pickle
import os

def save_nets(nets, name, directory='../res/'):
    #normal_nets = dict()
    # for gamma, [n1, n2] in nets.items():
    #    normal_nets[gamma] = [dict(n1), dict(n2)]

    with open(os.path.join(directory, f"{name}.pickle"), 'wb+') as f:
        pickle.dump(nets, f)

def transform_app_data(fname='../data/sample_10000_vt_mal_2017_2020_az_2020_benign_hashed_md5.csv'):
    sf_ds1_full = tc.SFrame.read_csv(fname, header=False, verbose=False)
    _=sf_ds1_full.rename({'X3':'fcount'})
    sf_ds1_full['apk']=sf_ds1_full['X1'].apply(lambda x: x.upper())
    sf_ds1_full['function'] = sf_ds1_full['X2'].apply(lambda x: x.upper())
    
    _=sf_ds1_full.remove_columns(['X1', 'X2'])
    
    print("Unique APK:", len(set(sf_ds1_full['apk'].unique())))
    print("Unique functions:", len(set(sf_ds1_full['function'].unique())))

    outname = fname.replace('data', 'binarydata').replace('.csv', '.sframe')
    sf_ds1_full.save(outname, format='binary')

def read_labels():
    apk_label_ds1_path = '../data/labels_vt_mal_2017_2020_az_2020_benign_hashed.csv'
    ds1_labels_df = pd.read_csv(apk_label_ds1_path)
    ds1_labels_df.set_index(ds1_labels_df.apk.apply(lambda x:x.upper()), inplace=True)
    ds1_labels_df.drop(labels='apk',axis=1, inplace=True)
#    ds1_labels_df.head(2)
    print("Got data: ", len(ds1_labels_df))
    return ds1_labels_df

def get_sample(mw, frac):
    apks = mw['apk'].unique()
    sample_apks = apks.sample(fraction=frac, seed=42)
    return mw.filter_by(sample_apks, column_name='apk')

def votes_from_list(li, classifier):
    classes = [classifier(a) for a in li]
    su = sum(classes)
    return len(classes)-su, su

def convert_to_voting(net, classifier):
    voting_network = defaultdict(lambda: [0, 0])

    for k, l in net.items():
        bi, mal = votes_from_list(l+[k], classifier)
        voting_network[k] = [bi, mal]

    return dict(voting_network)

def merge_voting_nets(nets, datas, gamma):
    dat = datas[0]
    for data in datas[1:]:
        dat = dat.append(data)

    nn = f_create_network(gamma=gamma, data=dat)
    
    # transfer the "votes" from original networks to just created new anchors
    targ = dict()
    for k, v in nn.items():
        for net in nets:
            if k in net:
                targ[k] = net.get(k)
                break

        for el in v:
            for net in nets:
                if el in net:
                    targ[k] = list(np.add(targ[k], net[el]))
                    break

    return targ

def old_f_create_network(data, gamma):
    apks = data['apk'].unique()
    k = apks.shape[0]
    sim_recom = tc.item_similarity_recommender.create(data, 
                                                      user_id='function', 
                                                      item_id='apk', 
                                                      similarity_type='jaccard', 
                                                      only_top_k=k, 
                                                      degree_approximation_threshold=15*4096, 
                                                      threshold=0.0, verbose=False)
    itms = sim_recom.get_similar_items(apks, k=k)
    # missing more "distant nodes", "not aggregating nodes"
    gw=itms[itms['score']>=1-gamma].groupby(key_column_names='apk', operations={'sims': agg.DISTINCT('similar')})
    
    ws = set(gw['apk'])
    net = dict()
    already_added = set()
    while len(ws)>0:
        w= ws.pop()

        simp = set(gw[gw['apk']==w]['sims'][0])
        simp = simp - already_added

        net[w] = list(simp)
        already_added.update(simp)
        already_added.add(w)

        ws = ws - simp
    
    # add solitary nodes & not-aggregating nodes
    if len(already_added)> 0:
        nds = apks.filter_by(list(already_added), exclude=True)
    else:
        nds = apks
        
    for n in nds:
        net[n] = []
        
    return net


def f_create_network(data, gamma):
    apks = data['apk'].unique()
    k = apks.shape[0]
    sim_recom = tc.item_similarity_recommender.create(data, 
                                                      user_id='function', 
                                                      item_id='apk', 
                                                      similarity_type='jaccard', 
                                                      only_top_k=k, 
                                                      degree_approximation_threshold=15*4096, 
                                                      threshold=0.0, verbose=False)
    itms = sim_recom.get_similar_items(apks, k=k)
    # missing more "distant nodes", "not aggregating nodes"
    # setup a list in form: apk [apk1, apk2,...]
    # all aggregated items will be closer than given radius, ie, they are def. not anchors
    # the list will be used to build network in a greedy way
    gw=itms[itms['score']>=1-gamma].groupby(key_column_names='apk', operations={'sims': agg.DISTINCT('similar')})
    
    ws = set(gw['apk'])
    anchors = list()
    already_added = set()
    # greedy building net
    # start with first row apk1, [apk2, apk3, apk4...], 
    # apk1 is added to network, 
    # remove all apks that are closer than radius (ie. all from the list) from the following rows
    while len(ws)>0:
        w= ws.pop()
        anchors.append(w)

        # all apks closer than given radius to this anchors
        simp = set(gw[gw['apk']==w]['sims'][0])
        simp = simp - already_added
            
        already_added.update(simp)
        already_added.add(w)

        # remove its aggregates 
        ws = ws - simp
    
    # move to nodes pairs that are at distance > radius
    # there should be no close nodes to that one, otherwise they
    # would appear in the previous loop
    if len(already_added)> 0:
        nds = apks.filter_by(list(already_added), exclude=True)
        
    # special case for gamma=0
    else:
        nds = apks
        
    
    anchors = anchors + list(nds)
    
    #following is equivallent to NN search on just created anchors:
    # done here to reuse existing recommender
    
    # recomendations excluding similarity between apks
    fitems = itms.filter_by(values=anchors, column_name='apk',exclude=True).filter_by(values=anchors,column_name='similar')
    
    gwrrd = fitems.groupby(key_column_names=['apk'], operations={'nn': tc.aggregate.ARGMAX('score', 'similar')})
    r = gwrrd.groupby(key_column_names=['nn'], operations={'aggregates': agg.DISTINCT('apk')})
    net = r.to_dataframe().set_index('nn').to_dict(orient='dict')['aggregates']

    # not used anchors added:
    net.update({an:[] for an in anchors if an not in net})
    return net


def partition_ndframe(nd, n_parts):
    rn = default_rng(42)
    permuted_indices = rn.permutation(len(nd))

    dfs = []
    for i in range(n_parts):
        dfs.append(nd[permuted_indices[i::n_parts]])
    return dfs

def dist_and_net(data, gamma):
    net = f_create_network(data=data, gamma=gamma)
    return net, data.filter_by(values=net.keys(), column_name='apk')

def convert_with_dist(pair, classifier):
    return convert_to_voting(pair[0], classifier=classifier), pair[1]

def make_and_merge(parts, labels, gamma):
    print(f"Starting network creation {gamma}")
    myfc = partial(dist_and_net, gamma=gamma)
    #with Pool() as p:
    aggregating_networks = [myfc(part) for part in parts]
    nets, _ = zip(*aggregating_networks)
    save_nets({gamma: nets}, f"{gamma}-tc-singleaggregating")


    classifier = lambda x: int(labels.loc[x]['malware_label'])
    myconv = partial(convert_with_dist, classifier=classifier)
    print("Starting voting conversion")
    #with Pool() as p:
    voting_networks = [myconv(pair) for pair in aggregating_networks]
    nets, _ = zip(*voting_networks)
    save_nets({gamma: nets}, f"{gamma}-tc-singlevoting")

    print("Merging")
    start = time.time()

    #nds = list(zip(voting_networks, parts))
    #part_merge = partial(mymerge, gamma=gamma)
    # while len(nds)>1:
    #    print(f"Hierarchical merging {len(nds)}")
    #    b = zip(nds[::2], nds[1::2])
    #    with Pool() as p:
    #        nds = p.map(part_merge,b)
    nets, datas = zip(*voting_networks)
    mv = merge_voting_nets(nets=nets, datas=datas, gamma=gamma)
    end = time.time()
    print(f"\tElapsed: {end-start}")
    return mv

if __name__=="__main__":
    mw = tc.load_sframe('../binarydata/funcs-encoded')
    mw.remove_column('fcount', inplace=True)
    #subsamp = get_sample(mw=mw, frac=0.2)
    subsamp = mw

    napks = subsamp['apk'].unique().to_numpy()
    test_size = 1000
    train, test = train_test_split(napks, test_size=test_size, random_state=42)
    np.save(f"../res/test-tc-{test_size}", test)

    parts = partition_ndframe(nd=train, n_parts=4)
    sparts = [subsamp.filter_by(values=part, column_name='apk') for part in parts]
    ftrain = subsamp.filter_by(values=train, column_name='apk')

    labels = pd.read_csv('../data/labels_encoded.csv', index_col=0)
    classifier = lambda x: int(labels.loc[x]['malware_label'])

    nets = dict()
    intervals = 18
    for gamma in tqdm([x * 1/intervals for x in range(0, intervals+1)]):
        print(f"Current {gamma=}")
        mv = make_and_merge(gamma=gamma, parts=sparts, labels=labels)
        
        print("Creating reference aggregating network")
        start = time.time()
        reference_agg = f_create_network(gamma=gamma, data=ftrain)
        reference_voting = convert_to_voting(reference_agg, classifier)
        end = time.time()
        #
        print(f"\tElapsed: {end-start}")
        save_nets({gamma: [reference_agg]}, f"{gamma}-tc-singlereferenceaggregating")
        
        nets[gamma] = [dict(mv), dict(reference_voting)]
    

        print(f"Anchor points: {len(mv.keys())}")
        if len(mv.keys()) == 1:
            break

    # save nets:
    save_nets(nets=nets, name=f"{len(train)}-tc-jaccard-votingnets")
    