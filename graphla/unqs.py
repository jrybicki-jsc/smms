from utils import load_functions_partition
import turicreate.aggregate as agg
import hashlib

def calc_hash(x, digits=12):
    return int(hashlib.sha1(f"{tuple(sorted(x))}".encode("utf-8")).hexdigest(), 16) % (10 ** digits)


if __name__ == "__main__":
    p = load_functions_partition(directory='../data/part-', name='0')
    ags = p.groupby(key_column_names='apk', operations={'afunc': agg.DISTINCT('function')})
    m = ags.apply(lambda x: calc_hash(x['afunc']))
    print(m.unique().shape, m.shape)