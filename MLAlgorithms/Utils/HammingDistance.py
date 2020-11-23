import numpy as np
from numba import njit

"""
Calculates the Hamming distance between two binary vectors

Will return -1 if the inputted data is bad

Args:
    arr1: array<int> | array<boolean>
    arr2: array<int> | array<boolean>

Returns:
    hammingDistance: int
"""
@njit
def calculateHammingDistance(arr1, arr2):

    if arr1 is None or arr2 is None: return -1
    if len(arr1) != len(arr2): return -1



    result = np.logical_xor(arr1, arr2)
    hammingDistance = np.count_nonzero(result)

    return hammingDistance
