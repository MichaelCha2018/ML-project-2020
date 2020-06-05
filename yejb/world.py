'''
    parse the hyperparameters
    and make the hyperparameters global
'''
import argparse
import torch
from pprint import pprint

def parse_args():
    parser = argparse.ArgumentParser(description="A simple DQN")
    parser.add_argument('--batchsize', type=int,default=32)
    parser.add_argument('--gamma', type=float,default=0.99)
    parser.add_argument('--buffersize', type=int,default=1000000)
    parser.add_argument('--startepoch', type=int,default=50000)
    parser.add_argument('--dqn_freq', type=int,default=4)
    parser.add_argument('--framelen', type=int,default=4)
    parser.add_argument('--dqn_updatefreq', type=int,default=10000)
    parser.add_argument('--lr', type=float,default=0.00025)
    parser.add_argument('--alpha', type=float,default=0.95)
    parser.add_argument('--eps', type=float,default=0.01)
    parser.add_argument('--imgDIM', type=int,default=84)
    parser.add_argument('--seed', type=int,default=2020)
    parser.add_argument('--timesteps', type=int,default=100000000)
    return parser.parse_args()
# ------------------------------------------------
args = parse_args()
ARGS = args
GPU = torch.cuda.is_available()
DEVICE = torch.device('cuda' if GPU else "cpu")
# ------------------------------------------------
# The above is annoation style
def Print(where, strs):
    print(f"\033[0;30;43mINFO\033[0m {where}")
    pprint(strs)
    print("% ---------------------------")
# ------------------------------------------------
Print('world', ARGS.__dict__)
# ------------------------------------------------
function_annoation_style = \
r"""@parameters

        arg1: what for
        arg2: what for
        ...
        
    @return
    
        r1: what for
        ...
    
    return r1, ...
"""