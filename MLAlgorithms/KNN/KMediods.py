from MLAlgorithms.Utils.DistanceMatrix import DistanceMatrix
import pandas as pd
import numpy as np
from numpy import genfromtxt
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


def KMediods(dataSet, classifier, discreteAttr, continAttr, predictionType, k, maxIter):
    # Pick k random cluster centers from the given dataspace
    mediods = _createMediods(dataSet, k)

    totalAttr = len(discreteAttr) + len(continAttr)
    percentDis = len(discreteAttr) / totalAttr
    percentCon = len(continAttr) / totalAttr

    iteration = 0

    # print(dataSet)
    distanceMatrix = genfromtxt('foo.csv', delimiter=',')
    # distanceMatrix = DistanceMatrix(dataSet, dataSet, continAttr, discreteAttr, percentCon, percentDis,
    #                                 predictionType, classifier)
    # print(distanceMatrix.distanceMatrix)
    # np.savetxt("foo.csv", distanceMatrix.distanceMatrix, delimiter=",")
    # print("done")
    mediodList = mediods.index

    while iteration < maxIter:
        flag = False
        # ============================================================================================= Assign each point to a cluster
        assignedClusters = {}

        for index, row in tqdm(enumerate(dataSet.to_numpy()), total=len(dataSet)):
            closestCentroidIndex = distanceMatrix[index][mediods.index].argmin()

            if mediods.iloc[[closestCentroidIndex]].index[0] not in assignedClusters:  # Assign the point to the centroid
                assignedClusters[mediods.iloc[[closestCentroidIndex]].index[0]] = []
            assignedClusters[mediods.iloc[[closestCentroidIndex]].index[0]].append(index)

        # ============================================================================================= calculate distortion
        initialDistortionSum = 0
        for mediodPoint in mediods.index:
            for dataPoint in assignedClusters[mediodPoint]:
                initialDistortionSum += (distanceMatrix[dataPoint][mediodPoint])**2

        print(initialDistortionSum)
        # ============================================================================================= Recalculate our centroids

        for mediodPoint in mediods.index:
            for dataPoint in dataSet.index:
                if mediodPoint == dataPoint:
                    continue
                else:
                    tempMediod = dataPoint
                    tempDataPoint = mediodPoint

                    newDistortionSum = 0
                    for mediodPointv in mediods.index:
                        if mediodPointv == mediodPoint:
                            mediodPointv = tempMediod
                        for dataPointv in assignedClusters[mediodPoint]:
                            if dataPointv == dataPoint:
                                dataPointv = tempDataPoint
                            newDistortionSum += (distanceMatrix[dataPointv][mediodPointv]) ** 2
                    if newDistortionSum <= initialDistortionSum:
                        flag = True
                        mediods.iloc[[mediodPoint]], dataSet.iloc[[dataPoint]] = dataSet.iloc[[tempDataPoint]], mediods.iloc[[tempMediod]]

        """
        mainDis = Calculate Distortion
        for mediod in mediods:
            for datapoint in DataPoints
                if datapoint -= mediod
                    continue
                swap mediod & datapoint
                
                loopDis = calculate Distortion 
                
                if loopDis <= mainDis
                    swap back
        
        
        If no change in mediods:
            end
        """

        # ============================================================================================= Check
        iteration += 1
        if not flag:
            break

    return pd.DataFrame(mediods).T


"""
What is Distortion?
    Sum over clusters k
        sum over points in cluster k
            ()
        

"""


# This function creates k centroids
def _createMediods(dataSet, k):
    seed = 69

    mediods = dataSet.sample(k, random_state=seed)
    # mediods.index = [f"centroid{i}" for i in range(0, k)]

    return mediods
