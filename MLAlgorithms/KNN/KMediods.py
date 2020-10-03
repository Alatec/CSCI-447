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


    distanceMatrix = DistanceMatrix(dataSet, dataSet, continAttr, discreteAttr, percentCon, percentDis, predictionType, classifier)

    medoidList = np.array(mediods.index)
    nonMedoidList = np.array(list(set(dataSet.index)^set(medoidList)))
    # nonMediodList = np.array(dataSet.index)

    mediodMatrix = np.zeros((len(dataSet), len(dataSet)), dtype=int)
    for medoid in medoidList:
        mediodMatrix[medoid][medoid] = medoid


    while iteration < maxIter:
        print(iteration)
        flag = False
        # ============================================================================================= Assign each point to a cluster
        assignedClusters = {}

        # # Assign each datapoint to a mediod
        for nonMedoid in nonMedoidList:
            closestMedoid = medoidList[0]
            for medoid in medoidList:
                if distanceMatrix.distanceMatrix[nonMedoid][medoid] < distanceMatrix.distanceMatrix[nonMedoid][closestMedoid]:
                    closestMedoid = medoid

            if closestMedoid not in assignedClusters:
                assignedClusters[closestMedoid] = []
            assignedClusters[closestMedoid].append(nonMedoid)



        # # ============================================================================================= calculate distortion
        initialDistortionSum = 0
        for mediodPoint in mediods.index:
            for dataPoint in assignedClusters[mediodPoint]:
                initialDistortionSum += (distanceMatrix.distanceMatrix[dataPoint][mediodPoint])**2

        print(initialDistortionSum)

        # # ============================================================================================= Recalculate our centroids
        for medoid in medoidList:
            for dataPoint in nonMedoidList:
                tempMedoid = dataPoint
                tempDataPoint = medoid

                initialDistortionSum = 0
                for mediodPoint in mediods.index:
                    for dataPoint in assignedClusters[mediodPoint]:
                        initialDistortionSum += (distanceMatrix.distanceMatrix[dataPoint][mediodPoint]) ** 2


        for medoid in medoidList:
            for dataPoint in nonMedoidList:
                tempMedoid = dataPoint
                tempDataPoint = medoid

                newDistortionSum = 0
                for medoid in medoidList:
                    for dataPoint in nonMedoidList:
                        tempMedoid = dataPoint
                        tempDataPoint = medoid

                        initialDistortionSum = 0
                        for mediodPoint in mediods.index:
                            for dataPoint in assignedClusters[mediodPoint]:
                                initialDistortionSum += (distanceMatrix.distanceMatrix[dataPoint][mediodPoint]) ** 2

        #
        # for mediodPoint in mediods.index:
        #     for dataPoint in dataSet.index:
        #         if mediodPoint == dataPoint:
        #             continue
        #         else:
        #             tempMediod = dataPoint
        #             tempDataPoint = mediodPoint
        #
        #             newDistortionSum = 0
        #             for mediodPointv in mediods.index:
        #                 if mediodPointv == mediodPoint:
        #                     mediodPointv = tempMediod
        #                 for dataPointv in assignedClusters[mediodPoint]:
        #                     if dataPointv == dataPoint:
        #                         dataPointv = tempDataPoint
        #                     newDistortionSum += (distanceMatrix[dataPointv][mediodPointv]) ** 2
        #             if newDistortionSum <= initialDistortionSum:
        #                 flag = True
        #                 mediods.iloc[[mediodPoint]], dataSet.iloc[[dataPoint]] = dataSet.iloc[[tempDataPoint]], mediods.iloc[[tempMediod]]

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

        # distanceMatrix.recalculateCentriods(mediods)

        # ============================================================================================= Check
        iteration += 1
        # if not flag:
        #     break

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
