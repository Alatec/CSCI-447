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
def KMeans(dataSet, classifier, discreteAttr, continAttr, predictionType, k, maxIter):
    #Pick k random cluster centers from the given dataspace
    centroids = _createCentroids(dataSet, k)
    totalAttr = len(discreteAttr) + len(continAttr)
    percentDis = len(discreteAttr)/totalAttr
    percentCon = len(continAttr)/totalAttr

    iteration = 0

    distanceMatrix = DistanceMatrix(dataSet, centroids, continAttr, discreteAttr, percentCon, percentDis, predictionType, classifier)


    while iteration < maxIter:

        flag = False
        # ============================================================================================= Assign each point to a cluster
        assignedClusters = {}

        for index, row in tqdm(enumerate(dataSet.to_numpy()), total=len(dataSet)):
            closestCentroidIndex = distanceMatrix.distanceMatrix[index].argmin()

            if centroids.iloc[[closestCentroidIndex]].index[0] not in assignedClusters: # Assign the point to the centroid
                assignedClusters[centroids.iloc[[closestCentroidIndex]].index[0]] = []
            assignedClusters[centroids.iloc[[closestCentroidIndex]].index[0]].append(index)

        # ============================================================================================= Recalculate our centroids

        for centroidId, assignedPoints in assignedClusters.items():
            for cAttr in continAttr:
                oldVal = centroids.loc[[centroidId]][cAttr].item()
                newVal = dataSet.iloc[assignedPoints][cAttr].mean()

                if abs(oldVal - newVal) > .01:
                    print(f"OLD: {oldVal} NEW: {newVal}")
                    flag = True

                centroids.at[centroidId, cAttr] = newVal

            for dAttr in discreteAttr:
                oldVal = centroids.loc[[centroidId]][dAttr].values
                newVal = dataSet.iloc[assignedPoints][dAttr].mode().head().values
                if oldVal[0] == newVal[0]:
                    continue
                else:
                    flag = True
                    centroids.at[centroidId, dAttr] = dataSet.iloc[assignedPoints][dAttr].mode().head().values[0]

        distanceMatrix.recalculateCentriods(centroids)

        # ============================================================================================= Check if they changed
        iteration += 1
        if not flag:
            break

    return pd.DataFrame(centroids).T


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

    return pd.DataFrame.from_dict(dict, orient="index")
