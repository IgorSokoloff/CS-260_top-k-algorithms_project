import numpy as np
from sklearn.model_selection import train_test_split
import time
import sys
import os
import argparse
from numpy.random import normal, uniform
from numpy.linalg import norm
import itertools


import math

import datetime
from IPython import display
from tqdm import tqdm
#import IPython

from contextlib import redirect_stdout
import shutil
import subprocess
#from tournament_numba import * 
from tournament import *

from numpy.random import RandomState

myrepr = lambda x: repr(round(x, 4)).replace('.',',') if isinstance(x, float) else repr(x) #for some methods we used diffrent rounding
intrepr = lambda x: int(x) if x.is_integer() else round(x,4)

parser = argparse.ArgumentParser(description='Run top-k algorithm')

parser.add_argument('--k', action='store', dest='k', type=int, default=1, help='Sparcification parameter')
parser.add_argument('--d', action='store', dest='d', type=int, default=100, help='dimentionality')
parser.add_argument('--n_samples', action='store', dest='n_samples', type=int, default=1, help='Stepsize factor')

parser.add_argument('--left', action='store', dest='left', type=int, default=-1, help='mean')
parser.add_argument('--right', action='store', dest='right', type=int, default=1, help='sigma')


args = parser.parse_args()
k = args.k
d = args.d
n_samples = args.n_samples
left = args.left
right = args.right

#test
'''
k = 10
d = 100
n_samples = 10
left = -10
right = 10
'''
project_path = os.getcwd() + "/"

PRINT_EVERY = 10

algorithm_name = "tournament"
distribution_family = "uniform"
experiment_type = "synthetic"

rs = RandomState(12345)
tournament = TournamentTopK()

distribution = distribution_family+f"_l-{myrepr(left)}_r-{myrepr(right)}"
experiment = '{0}_{1}'.format(experiment_type, algorithm_name)
logs_path = project_path + "logs/logs_{0}/".format(experiment)

print (experiment)
#a global folder to store time complexity results
if not os.path.exists(project_path + "logs/"):
    os.makedirs(project_path + "logs/")

#a folder to store time complexity results of the experiments groupped by experiment_type, algorithm_name and distribution
if not os.path.exists(logs_path):
    os.makedirs(logs_path)

tc_hist = np.zeros(n_samples)

for i_s in range(n_samples):
    sample_vector =  rs.uniform(low=left, high=right, size=d)
    topk_knuth, numberOfComparisons = tournament.getTopK(list(sample_vector), k)
    tc_hist[i_s] = numberOfComparisons
    if i_s % PRINT_EVERY == 0:
        print (i_s)
tc_hist_name = 'TCH_prior-{0}_n-{1}_d-{2}_k-{3}.npy'.format(distribution, n_samples, d, k)

#code saving procedure

np.save(logs_path + tc_hist_name, tc_hist)
