from numpy.random import default_rng
import random

def join_tables(funcs, labels):
    lapns = labels.set_index('apn')
    gj = lapns.join(funcs)
    gj.dropna(inplace=True)
    gj['mal'] = gj.malware_label.astype(int)
    return gj.drop('malware_label', axis=1)

def partition_dataframe(df, n_parts):
    rn = default_rng(42)
    permuted_indices = rn.permutation(len(df))

    dfs = []
    for i in range(n_parts):
        dfs.append(df.iloc[permuted_indices[i::n_parts]])
    return dfs

# sample number of apps, and included all their functions in the returned dataset
def mysample(v, sample_size):
    r = random.sample(list(v.apn.unique()), k=sample_size)
    # apns = v.apn.unique sample(sample_size, random_state=42)
    return v[v.apn.isin(r)][['apn', 'nf']]


