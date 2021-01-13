import random
import pandas as pd
import numpy as np
#import graphlab as tc
import turicreate as tc
from tqdm.notebook import tqdm
import turicreate.aggregate as agg
import time


def generate_randoms(max_apk, func_nr, max_func):
    return {n: set(random.sample(range(1, func_nr), random.randint(1,max_func))) for n in range(max_apk)}

def denorm(cons):
    dat2 = {'apk': [], 'function': []}
    for apk, lsts in cons.items():
        for fncs in lsts:
            dat2['apk'].append(apk)
            dat2['function'].append(fncs)
    return dat2

def jaccard(apid1: int, apid2: int, funcs) -> float:
    p1 = funcs[apid1]
    p2 = funcs[apid2]

    return 1 - len(p1 & p2)/len(p1|p2) 

def ja(p1, p2):
    return 1 - len(p1 & p2)/len(p1|p2)

def create_aggregating_net(gamma,apns, distance):
    net = dict()

    for a in apns:
        insert = True
        for n in net.keys():
            if distance(a, n) <= gamma:
                insert = False
                net[n].append(a)
                break  
        if insert:
            net[a] = list()

    return net

def f_create_network(data, gamma):
    apks = data['apk'].unique()
    k = apks.shape[0]
    sim_recom = tc.item_similarity_recommender.create(data, 
                                                      user_id='function', 
                                                      item_id='apk', 
                                                      similarity_type='jaccard', 
                                                      only_top_k=k, verbose=False)
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

if __name__=="__main__":
    gamma = .7
    for size in range(1000, 10000, 2000):
        sms = generate_randoms(size, size//10, size//20)

        start = time.time()
        n=create_aggregating_net(gamma=gamma, apns=sms.keys(), distance=lambda x,y: ja(sms[x], sms[y]))
        di_dura = time.time() - start 

        art = tc.SFrame(data=denorm(sms))
        start = time.time()
        n2=f_create_network(data=art, gamma=gamma)
        tc_dura = time.time() - start
        
        print(f"{size}\t{di_dura}\t{tc_dura}")