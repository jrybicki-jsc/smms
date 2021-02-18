#!/usr/bin/env python 

import argparse
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Net spliter utility.')
    parser.add_argument('--net-file', help='name of the network file', required=True)
    parser.add_argument('--out', help='template name of the output', default='res/out_{}.pickle')
    args = parser.parse_args()
    print(args)

    with open(args.net_file, 'rb') as f:
        nets = pickle.load(f)
    
    a = {k: v[0] for k,v in nets.items()}
    b = {k: v[1] for k,v in nets.items()}

    with open(args.out.format(1), 'wb+') as f:
        pickle.dump(a, f)

    with open(args.out.format(2), 'wb+') as f:
        pickle.dump(b, f)
