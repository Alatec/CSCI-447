import numba
from numba import int32, float32, float64 
import numpy as np

spec = [
    ('test', float64[:,:]),
    ('train', float64[:,:])
]

@numba.njit(float64[:,:](float64[:,:],float64[:,:]))
def calculate_euclid_distances(test, train):
    distances = np.zeros((test.shape[0], train.shape[0]), dtype=np.float64)

    for i in range(test.shape[0]):
        for j in range(train.shape[0]):
            distances[i,j] = ((test[i]-train[j])**2).sum()

    return distances