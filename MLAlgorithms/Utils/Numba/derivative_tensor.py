import numba
from numba import int32, float32, float64, jit
import numpy as np


# Train is centroid

@jit(nopython=True)
def calculate_derivative_tensor(derivatives, weight_matrices):
    output_tensor = np.ones((derivatives.shape[0], weights.shape[0], weights.shape[1]))

    for i in range(test.shape[0]):
        for j in range(train.shape[0]):
            distances[i,j] = ((test[i]-train[j])**2).sum()

    return distances