from collections import defaultdict
from appknn import create_net, create_aggregating_net
import numpy as np

def naive_merge(n1, n2):
    # takes keys from both networks if key overlap the aggregates are merged

    nx = {**n1, **n2}
    for key in nx.keys():
        if key in n1:
            nx[key] = list(set(nx[key]+n1[key]))
    return nx


def key_based_merg(n1, n2, distance, gamma):
    # takes only keys that are at least gamma from each other, for keys closer than gamma, their aggregats are merged

    nx = {**n1}
    for k, l in n2.items():
        idf = True
        for k2 in nx.keys():
            if distance(k, k2) < gamma:
                nx[k2] += [li for li in l]
                idf = False
                break

        if idf:
            if k not in nx:
                nx[k] = []
            nx[k] += [li for li in l]

    return nx


def net_based_merge(n1, n2, distance, gamma):
    # calculates net over point nets and use it as keys of the merged network (similar/same? to the key-based)

    nn = create_aggregating_net(gamma=gamma,
                                apns=list(n1.keys())+list(n2.keys()),
                                distance=distance)
    targ = defaultdict(list)

    for k, v in nn.items():
        targ[k] = n1.get(k, []).copy()
        targ[k] += n2.get(k, []).copy()
        for el in v:
            targ[k] += n1.get(el, []).copy()
            targ[k] += n2.get(el, []).copy()
            targ[k].append(el)

    return targ


def net_based_multi_merge(nets, distance, gamma):
    apns = []
    for net in nets:
        apns += list(net.keys())

    nn = create_aggregating_net(gamma=gamma,
                                apns=apns,
                                distance=distance)
    targ = defaultdict(list)

    for k, v in nn.items():
        for net in nets:
            targ[k] += net.get(k, []).copy()

        for el in v:
            for net in nets:
                targ[k] += net.get(el, []).copy()
            targ[k].append(el)

    return targ


def merge_voting_nets(nets, distance, gamma):
    apns = []
    for net in nets:
        apns += list(net.keys())

    nn = create_aggregating_net(gamma=gamma,
                                apns=apns,
                                distance=distance)
    
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
