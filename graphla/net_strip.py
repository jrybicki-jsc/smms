# striper
import argparse

from utils import load_net
from grapm import save_nets

def strip_net(net):
    return {a:[] for a in net}

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Strip net')
    parser.add_argument('--net', help='net file', required=True)
    parser.add_argument('--output', help='output path', required=False)
    args = parser.parse_args()
    g, n = load_net(args.net)
    st = strip_net(n)
    if args.output:
        save_nets({g: [st]}, "stripped",  directory=args.output)
    else:
        fname = args.net.replace('.pickle', '-stripped')
        save_nets({g: [st]}, fname,  directory='')
