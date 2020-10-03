from MLAlgorithms.Utils.DistanceMatrix import DistanceMatrix
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


def KMediods(dataSet, classifier, discreteAttr, continAttr, predictionType, k, maxIter):
    # Pick k random cluster centers from the given dataspace
    mediods = _createMediods(dataSet, k)

    totalAttr = len(discreteAttr) + len(continAttr)
    percentDis = len(discreteAttr) / totalAttr
    percentCon = len(continAttr) / totalAttr

    iteration = 0


    distanceMatrix = DistanceMatrix(dataSet, dataSet, continAttr, discreteAttr, percentCon, percentDis, predictionType, classifier)

    medoidList = np.array(mediods.index)
    nonMedoidList = np.array(list(set(dataSet.index)^set(medoidList)))


    mediodMatrix = np.zeros((len(dataSet), len(dataSet)), dtype=int)
    for medoid in medoidList:
        mediodMatrix[medoid][medoid] = medoid


    assignedClusters = {}
    while iteration < maxIter:
        oldMedoidList = np.copy(medoidList)

        # ============================================================================================= Assign each point to a cluster

        if assignedClusters == {}:
            for nonMedoid in nonMedoidList:
                closestMedoid = medoidList[0]
                for medoid in medoidList:
                    if distanceMatrix.distanceMatrix[nonMedoid][medoid] < distanceMatrix.distanceMatrix[nonMedoid][closestMedoid]:
                        closestMedoid = medoid

                if closestMedoid not in assignedClusters:
                    assignedClusters[closestMedoid] = []
                assignedClusters[closestMedoid].append(nonMedoid)
            for medoid in medoidList:
                if medoid not in assignedClusters:
                    assignedClusters[medoid] = []



        # # ============================================================================================= calculate distortion

        initialDistortionSum = 0
        for mediodPoint in medoidList:
            for dataPoint in assignedClusters[mediodPoint]:
                initialDistortionSum += (distanceMatrix.distanceMatrix[dataPoint][mediodPoint])**2


        # # ============================================================================================= Recalculate our centroids

        for medoid, medoidRow in enumerate(medoidList):

            for dataPoint, dataRow in enumerate(dataSet.index):
                if medoid == dataPoint:
                    continue
                else:
                    tempMedoid = dataRow
                    tempDataPoint = medoidRow

                    newDistortionSum = 0
                    for mediodPointP in medoidList:

                        if mediodPointP == medoid:
                            mediodPointP = tempMedoid


                        for dataPointP in assignedClusters[medoidRow]:
                            if dataPointP == dataPoint:
                                dataPointP = tempDataPoint

                            newDistortionSum += (distanceMatrix.distanceMatrix[dataPointP][mediodPointP]) ** 2

                    if newDistortionSum > initialDistortionSum and tempMedoid not in medoidList:

                        assignedClusters[tempMedoid] = assignedClusters[medoidRow]
                        del assignedClusters[medoidRow]

                        medoidList[medoid] = tempMedoid
                        medoidRow = tempMedoid

        # ============================================================================================= Check
        iteration += 1
        if np.array_equal(oldMedoidList, medoidList):
            break

    print(dataSet.loc[medoidList])
    return dataSet.loc[medoidList]


# This function creates k centroids
def _createMediods(dataSet, k):
    seed = 69


    mediods = dataSet.sample(k, random_state=seed)

    return mediods
