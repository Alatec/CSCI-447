from MLAlgorithms.Utils.DistanceValueMetric import DistanceValueMetric
from MLAlgorithms.Utils.NumbaFunctions import calculate_euclid_distances
import pandas as pd
import numpy as np
from tqdm import tqdm


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
    #Pick k random cluster centers from the given dataspace
    centroids = _createCentroids(dataSet, k)
    dataMetric = DistanceValueMetric(dataSet, classifier, discreteAttr,
                                     predictionType)
    # calculatedContinDistance = calculate_euclid_distances(, )

    totalAttr = len(discreteAttr) + len(continAttr)
    percentDis = len(discreteAttr)/totalAttr
    percentCon = len(continAttr)/totalAttr

    iteration = 0

    attrDict = {}
    attrArray = dataSet.columns.to_numpy()

    for i in range(0, len(attrArray)):
        attrDict[attrArray[i]] = i

    while iteration < maxIter:
        print(iteration)
        flag = False
        # ============================================================================================= Assign each point to a cluster
        assignedClusters = {}

        for index, row in tqdm(enumerate(dataSet.to_numpy()), total=len(dataSet)):
            lowestDict = {}
            for cent, centVal in centroids.items():
                totalDistance = 0
                totalDistance += percentDis*dataMetric.calculateDistance(centVal, row, attrDict) # Calculate every categorical

                list = []     # Calculate every continuous data
                for attr in continAttr:
                    list.append(row[attrDict[attr]])
                continAttrMatrix = calculate_euclid_distances(np.asarray(list, dtype=np.float64), centVal[continAttr].to_numpy(dtype=np.float64))
                totalDistance += percentCon*continAttrMatrix.sum().sum()

                lowestDict[cent] = totalDistance

            lowestCentroid = min(lowestDict.items(), key=lambda x: x[1]) # Return the closest centroid to the point

            if lowestCentroid[0] not in assignedClusters: # Assign the point to the centroid
                assignedClusters[lowestCentroid[0]] = []
            assignedClusters[lowestCentroid[0]].append(index)

        # ============================================================================================= Recalculate our centroids

        for centroidId, assignedPoints in assignedClusters.items():
            for cAttr in continAttr:
                oldVal = centroids[centroidId][cAttr]
                newVal = dataSet.iloc[assignedPoints][cAttr].mean()
                if abs(oldVal - newVal) < .01:
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
