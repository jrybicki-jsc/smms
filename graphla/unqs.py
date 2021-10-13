from utils import load_functions_partition
import turicreate.aggregate as agg
import hashlib
import argparse

def calc_hash(x, digits=12):
    return int(hashlib.sha1(f"{tuple(sorted(x))}".encode("utf-8")).hexdigest(), 16) % (10 ** digits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate duplicates')
    parser.add_argument('--functions', help='name of the functions directory', required=True)
    args = parser.parse_args()

    print(f"Loading {args.functions}")
    p = load_functions_partition(directory=args.functions, name='')
    ags = p.groupby(key_column_names='apk', operations={'afunc': agg.DISTINCT('function')})
    m = ags.apply(lambda x: calc_hash(x['afunc']))
    print("data size, unique apks, all apks")
    print(p.shape, m.unique().shape, m.shape)