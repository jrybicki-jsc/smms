#!/usr/bin/env python3
#print nets stats
import glob
from utils import load_net

if __name__=="__main__":
    
    for fname in glob.glob('*.pickle'):
        try:
            g2, net = load_net(fname)
            print(f"{fname} gamma={g2} size: {len(net)}")
        except:
            print(f"Unable to load net from {fname}")

        