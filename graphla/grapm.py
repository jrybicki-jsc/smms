import pandas as pd
import numpy as np
#import graphlab as gl
import turicreate as tc
import turicreate.aggregate as agg
from tqdm.notebook import tqdm
import time
from tqdm import tqdm

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
    
def get_similar_apks(apk):
    similar_items = sim_recom.get_similar_items([apk], k=k)['similar', 'score']
    similar_items.materialize()
    return similar_items

def get_jaccard_sim(apk1, apk2):
    similar_to_apk1 = get_similar_apks(apk1)
    try:
        return similar_to_apk1[similar_to_apk1['similar']==apk2]['score'][0]
    except:
        return 0

def get_recommender(data):
    k = len(data['apk'].unique())
    return k, tc.item_similarity_recommender.create(data, 
                                                 user_id='function',
                                                 item_id='apk',
                                                 similarity_type='jaccard',
                                                 verbose=False, only_top_k=k)
    
def jaccard_dist(apk1, apk2):
    return 1 - get_jaccard_sim(apk1, apk2)

def aio_distance(apk1, apk2, recommender, k):
    similar_items = recommender.get_similar_items([apk1], k=k)['similar', 'score']
    similar_items.materialize()
    
    try:
        return 1-similar_items[similar_items['similar']==apk2]['score'][0]
    except:
        return 1
    
def classifier(apk, labels):
    return [[0, 1], [1, 0]][int(labels.loc[apk]['malware_label'])]

def setup_rec(data):
    apks = data['apk'].unique()
    k = len(apks)
    sim_recom = tc.item_similarity_recommender.create(data, 
                                                      user_id='function',
                                                      item_id='apk',
                                                      similarity_type='jaccard',
                                                      verbose=False,only_top_k=k)
    return apks, k, sim_recom

def create_network_alt(data, gamma, apks, k, sim_recom):
    itms = sim_recom.get_similar_items(apks, k=k)

    # potentially loosing some anchors here? 
    gw=itms[itms['score']>=1-gamma] 
    gw = gw.groupby(key_column_names='apk', operations={'sims': agg.DISTINCT('similar')})
    ws = set(gw['apk'])

    net = dict()
    already_added = set()
    while len(ws)>0:
        w= ws.pop()

        simp = set(gw[gw['apk']==w]['sims'][0])
        simp = simp - already_added

        net[w] = simp
        already_added.update(simp)
        already_added.add(w)

        ws = ws - simp
    return net

def get_sarray_parts(sa, num_parts, size):
    permuted_indices = np.random.permutation(len(sa))
    return [[sa[permuted_indices[j]] for j in range(i,i+size)] for i in range(0, size*num_parts, size)]

if __name__=="__main__":
    mw = tc.load_sframe('../binarydata/sample_10000_vt_mal_2017_2020_az_2020_benign_hashed_md5.sframe')
    subsamp = get_sample(mw=mw, frac=0.1)
    #sample_apks = subsamp['apk'].unique()
    #labels = read_labels()
    #k, rec = get_recommender(subsamp)
    #distance = lambda x,y: aio_distance(x,y, rec, k)
    #clas = lambda x: classifier(x, labels)
    tts = dict()
    apks, k, sim_recom = setup_rec(subsamp)

    for gamma in tqdm([0.0, 0.2, 0.5, 0.8]):
        print("Starting network calculation")
        start = time.time()
        #net2 = create_voting_net_alt(gamma=gamma, apns=sample_apks, classifier=clas, distance=distance)
        nn = create_network_alt(data=subsamp, gamma=gamma, apks=apks, k=k, sim_recom=sim_recom)
        tts[gamma] = time.time() - start
        print(f"Network calculation for {gamma=} took: {tts[gamma]}")

