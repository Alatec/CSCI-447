import pandas as pd
import numpy as np


"""
Clusters the data to be used for the various KNN algorithms

Ensure that the dataframe has been preprocessed before being inputted
Args: 
    dataSet: Pandas DataFrame
    k: int
    
Returns:
    clusteredData: Panadas DataFrame
"""
def KMeans(dataSet, k):
    # Pick k random cluster centers from the given dataspace
    centroids = _createCentroids(dataSet, k)
    flag = True

    while flag:
        for row in dataSet:
            pass

    """
    while our centroids haven't change:
    |   for row in data set:
    |    |   calculate distance between row and centroid
    |    |   assign the row to the closest centroid
    |   edit our centroids to be the mean of each assigned row
    return centroids
    
    Use a flag in the centroid mean calculation. 
        -> At the start of each loop, set the flag to false
        -> If there is a centroid, set the flag to true.
            -> Create a if statement so we don't reassign multiple times
            
    When calculating distance, use an if else to determine which function
        -> 
    
    
    """


    return None


def _calculateMean():
    pass

# This function creates k centroids
def _createCentroids(dataSet, k):
    seed = 69
    dict = {}

    # Populate the dictionary with k centroids
    for i in range(k):
        np.random.seed(seed)
        seed += 1

        key = "centroid" + str(i)
        val = pd.DataFrame([np.random.choice(dataSet[i], 1)[0] for i in dataSet.columns]).T
        val.columns = dataSet.columns

        dict[key] = val

    return dict
