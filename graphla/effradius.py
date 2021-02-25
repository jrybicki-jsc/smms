import turicreate as tc
from tqdm import tqdm
import turicreate.aggregate as agg
import pickle

def setup_items(data):
    print('Setting up items')
    apks = mw['apk'].unique()
    k = apks.shape[0]
    sim_recom = tc.item_similarity_recommender.create(mw, 
                                                      user_id='function', 
                                                      item_id='apk', 
                                                      similarity_type='jaccard', 
                                                      only_top_k=k,
                                                      degree_approximation_threshold = 15*4096,
                                                      threshold=0.0, verbose=False)
    items =sim_recom.get_similar_items(apks, k=int(k))

    items['distance'] = 1.0 - items['score']

    items = items.remove_columns(column_names=['score', 'rank'])
    return items

def get_over_distances(net, items, gamma):
    for_gam_di = items[items['distance']>gamma].filter_by(values=list(net.keys()), column_name='apk')
    res = dict()
    for key, aggs in tqdm(net.items()):
        loc_dat = for_gam_di[for_gam_di['apk']==key].filter_by(values=aggs, column_name='similar')
        max_dst = loc_dat['distance'].max()
        if max_dst:
            row = loc_dat[loc_dat['distance']==max_dst][0]['similar']
            res[key] = [max_dst, row]
    return res            



if __name__=="__main__":
    mw = tc.load_sframe('../binarydata/funcs-encoded')
    mw = mw.remove_column('fcount', inplace=True)
    
    net_file_name = '../res/merged-th15_merged.pickle'
    print(f"Reading in nets from {net_file_name}")
    with open(net_file_name, 'rb') as f:
        allthenet = pickle.load(f)

    items = setup_items(data=mw)

    distances = dict()
    for gamma, net in tqdm(allthenet.items()):
        distances[gamma] = get_over_distances(net, items, gamma=gamma)

        # snapshot each round
        with open('distances.pickle', 'wb+') as f:
            pickle.dump(distances, f)
   
    