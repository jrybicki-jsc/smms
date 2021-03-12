from grapm import f_create_network
from collections import defaultdict
import glob
import turicreate as tc
import turicreate.aggregate as agg
import pickle
from tqdm import tqdm

def get_networks():
    all_merged = dict()
    for an in glob.glob('../res/*-tc-singleaggregating.pickle'):
        w =an.split('-')[0][len('../res/'):]
        gamma = float(w)

        with open(an, 'rb') as f:
            net = pickle.load(f)
        print(gamma, len(net[gamma]))
        all_merged[gamma] = net[gamma]
        
    return all_merged


def net_based_multi_merge(nets, datas, gamma):
    dat = datas[0]
    for data in datas[1:]:
        dat = dat.append(data)

    nn = f_create_network(gamma=gamma, data=dat)
    targ = defaultdict(list)

    for k, v in nn.items():
        for net in nets:
            targ[k] += net.get(k, []).copy()

        for el in v:
            for net in nets:
                targ[k] += net.get(el, []).copy()
            targ[k].append(el)

    return targ

if __name__ == "__main__":
    mw = tc.load_sframe('../binarydata/funcs-encoded')
    mw = mw.remove_column('fcount', inplace=True)

    all_merged = get_networks()
    res = dict()
    for gamma, nets in tqdm(all_merged.items()):
        dats = [mw.filter_by(values=net.keys(), column_name='apk')  for net in nets]
        mer = net_based_multi_merge(nets=nets, datas = dats, gamma=gamma)
        res[gamma] = mer
        # snapshot each round
        with open('res.pickle', 'wb+') as f:
            pickle.dump(res, f)
