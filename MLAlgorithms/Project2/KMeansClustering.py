from MLAlgorithms.Utils.DistanceValueMetric import DistanceValueMetric
from MLAlgorithms.Utils.NumbaFunctions import calculate_euclid_distances
import pandas as pd
import numpy as np


"""
data.getDataSet(), data.getDataClass(), data.getDescreteAttributes(), data.getPredictionType()
Clusters the data to be used for the various KNN algorithms

Ensure that the dataframe has been preprocessed before being inputted
Args: 
    dataSet: Pandas DataFrame
    classfier: String
    discreteAttr: List<String>
    predictionType: String
    k: int
    
Returns:
    clusteredData: Panadas DataFrame
"""
def KMeans(dataSet, classifier, discreteAttr, continAttr, predictionType, k):
    dataMetric = DistanceValueMetric(dataSet, classifier, discreteAttr,
                                     predictionType)

    #Pick k random cluster centers from the given dataspace
    centroids = _createCentroids(dataSet, k)
    flag = True

    while flag:
        assignedClusters = {}
        for index, row in dataSet.iterrows():
            lowestDict = {}
            for cent, centVal in centroids.items():
                totalDistance = 0
                totalDistance += dataMetric.calculateDistance(centVal, row) # Calculate every categorical
                for cont in continAttr:
                    totalDistance += calculate_euclid_distances(row[cont], centVal)
        break


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
        val = pd.Series([np.random.choice(dataSet[i], 1)[0] for i in dataSet.columns]).T
        val.index = dataSet.columns

        dict[key] = val

    return dict
