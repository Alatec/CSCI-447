from MLAlgorithms.Utils.DistanceValueMetric import DistanceValueMetric
from MLAlgorithms.Utils.NumbaFunctions import calculate_euclid_distances
import pandas as pd
import numpy as np
import math as m


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
def KMeans(dataSet, classifier, discreteAttr, continAttr, predictionType, k, maxIter):
    dataMetric = DistanceValueMetric(dataSet, classifier, discreteAttr,
                                     predictionType)

    totalAttr = len(discreteAttr) + len(continAttr)
    percentDis = len(discreteAttr)/totalAttr
    percentCon = len(continAttr)/totalAttr

    #Pick k random cluster centers from the given dataspace
    centroids = _createCentroids(dataSet, k)
    iteration = 0

    while iteration < maxIter:
        print(iteration)
        flag = False
# ============================================================================================= Assign each point to a cluster
        assignedClusters = {}
        for index, row in dataSet.iterrows():
            lowestDict = {}
            for cent, centVal in centroids.items():
                totalDistance = 0
                totalDistance += percentDis*dataMetric.calculateDistance(centVal, row) # Calculate every categorical
                continAttrMatrix = calculate_euclid_distances(row[continAttr].to_numpy(dtype="float64"), centVal[continAttr].to_numpy(dtype="float64"))
                continAttrFrame = pd.DataFrame(continAttrMatrix, continAttr, columns=continAttr)

                for cont in continAttr:
                    totalDistance += percentCon*continAttrFrame[cont][cont]

                lowestDict[cent] = totalDistance

            lowestCentroid = min(lowestDict.items(), key=lambda x: x[1])

            if lowestCentroid[0] not in assignedClusters:
                assignedClusters[lowestCentroid[0]] = []
            assignedClusters[lowestCentroid[0]].append(index)

# ============================================================================================= Recalculate our centroids

        for centroidId, assignedPoints in assignedClusters.items():
            for cAttr in continAttr:
                oldVal = centroids[centroidId][cAttr]
                newVal = dataSet.iloc[assignedPoints][cAttr].mean()
                if round(oldVal, 2) != round(newVal, 2):
                    flag = True
                centroids[centroidId][cAttr] = dataSet.iloc[assignedPoints][cAttr].mean()
            for dAttr in discreteAttr:
                oldVal = centroids[centroidId][dAttr]
                newVal = dataSet.iloc[assignedPoints][dAttr].mode().head().values
                if oldVal in newVal:
                    continue
                else:
                    flag = True
                    centroids[centroidId][dAttr] = dataSet.iloc[assignedPoints][dAttr].mode().head().values[0]

# ============================================================================================= Check if they changed
        iteration += 1
        if not flag:
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


    return pd.DataFrame(centroids).T


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
