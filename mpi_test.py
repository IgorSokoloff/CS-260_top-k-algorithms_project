from mpi4py import MPI
import numpy as np
from numpy.random import RandomState
from knuth_tournament.tournament import *

comm = MPI.COMM_WORLD
numOfWorkers = comm.Get_size()
rank = comm.Get_rank()

# 100 fixed seeds
seed = [42710, 20922, 95205, 34642, 40496, 48492, 32337, 48259, 83023,
       34473, 67642, 36856, 65281, 21858, 19010, 13632, 96642, 57753,
       84157, 53282, 41811, 44199, 90512, 44677, 96098, 78918, 52224,
       60217, 42397, 67461, 92649, 81195, 98683, 45370, 59958, 78805,
       93434, 89907, 33893, 76972, 28628, 73764, 21813, 88864, 31586,
       32365, 52275, 84206, 72030, 55327, 58580, 74869, 27426, 92806,
       54652, 23855, 81079, 23038, 72971, 75018, 28079, 83508, 21799,
       83420, 85260, 45813, 44349, 83063, 33089, 45654, 60944, 92720,
       26427, 87913, 35043, 43284, 22041, 94621, 74432, 97336, 12794,
       58293, 47644, 32575, 43923, 75956, 78123, 54400, 38063, 96111,
       35492, 72585, 69170, 62859, 94657, 28409, 43467, 94969, 40770,
       55856]

# Use a known seed for each worker. This will only work up to rank 99 
rs = RandomState(seed[rank])

# k = args.k
# d = args.d
# n_samples = args.n_samples
# mean = args.mean
# sigma = args.sigma
# pivot = args.pivot

k = 1000
d = 100000
mean = 0
sigma = 1

# ==========================================
# Define the task for each worker
# ==========================================
sample_vector =  rs.normal(loc=mean, scale=sigma, size=d)

tournament = TournamentTopK()
sendTopK, sendNumOfComparisons = tournament.getTopK(sample_vector, k)
sendNumOfComparisons = np.array(sendNumOfComparisons)

# print("Rank {}, sendTopK sent:\n{}".format(rank, sendTopK))

# ==========================================
# Define the task for worker (rank 0) that concatenate outputs from workers 
# ==========================================
recvTopK = None
recvNumOfComparisons = None
if rank == 0:
    # Define buffer size
    recvTopK = np.empty([numOfWorkers, d], dtype='float64')
    recvNumOfComparisons = np.empty([numOfWorkers, 1], dtype='i')

# Concatenate outputs for workers
comm.Gather(sendTopK, recvTopK, root=0)
comm.Gather(sendNumOfComparisons, recvNumOfComparisons, root=0)

if rank == 0:
    # np.set_printoptions(precision=1)
    # print("\nRank {}, recvTopK received:\n{}".format(rank, recvTopK))
    print('\nRank {}, recvNumOfComparisons received:\n{}'.format(rank, recvNumOfComparisons))
        
    #TODO: save recvNumOfComparisons to a file

