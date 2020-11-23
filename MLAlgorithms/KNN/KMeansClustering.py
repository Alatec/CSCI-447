from MLAlgorithms.Utils.DistanceMatrix import DistanceMatrix
import pandas as pd
import numpy as np
from tqdm import tqdm


"""
Clusters the data to be used for the various KNN algorithms with centroids

Ensure that the DataFrame has been preprocessed before being inputted

Args: 
    dataSet: Pandas DataFrame
    classfier: String
    discreteAttr: List<String>
    continAttr: List<String>
    predictionType: String
    k: int
    maxUter: int
    
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

    # Create a matrix of scalar values between each data point and centroid
    distanceMatrix = DistanceMatrix(dataSet, centroids, continAttr, discreteAttr, percentCon, percentDis, predictionType, classifier)


    # Recalculate centroids until no change in the centroids
    while iteration < maxIter:
        print(iteration)
        flag = False

        # =================================================================== Assign each point to a cluster
        assignedClusters = {}

        for index, row in tqdm(enumerate(dataSet.to_numpy()), total=len(dataSet)):
            closestCentroidIndex = distanceMatrix.distanceMatrix[index].argmin()

            # Assign the point to the centroid
            if centroids.iloc[[closestCentroidIndex]].index[0] not in assignedClusters:
                assignedClusters[centroids.iloc[[closestCentroidIndex]].index[0]] = []
            assignedClusters[centroids.iloc[[closestCentroidIndex]].index[0]].append(index)

        # ================================================================ Recalculate our centroids

        for centroidId, assignedPoints in assignedClusters.items():
            for cAttr in continAttr:
                oldVal = centroids.loc[[centroidId]][cAttr].item()
                newVal = dataSet.iloc[assignedPoints][cAttr].mean()

                # If we have a change in our continuous values, we need to iterate again
                if abs(oldVal - newVal) > .01:
                    print(f"OLD: {oldVal} NEW: {newVal}")
                    flag = True

                centroids.at[centroidId, cAttr] = newVal

            for dAttr in discreteAttr:                  # If we have a new categorical value, we need to iterate again
                oldVal = centroids.loc[[centroidId]][dAttr].values
                newVal = dataSet.iloc[assignedPoints][dAttr].mode().head().values
                if oldVal[0] == newVal[0]:
                    continue
                else:
                    flag = True
                    centroids.at[centroidId, dAttr] = dataSet.iloc[assignedPoints][dAttr].mode().head().values[0]

        # ============================================= If the flag has been tripped (centroids changed), iterate again
        iteration += 1
        if not flag:
            break

        # Otherwise, recalculate the centroids
        distanceMatrix.recalculateCentriods(centroids)

    return pd.DataFrame(centroids)


# This function creates k centroids that have values pulled from the data space
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
