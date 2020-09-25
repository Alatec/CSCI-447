import pandas as pd
import numpy as np

def KMeans(dataSet, k):
    # Pick k random custer centers from our dataspace
    centroids = _createCentroids(dataSet, k)

    for key, val in centroids.items():
        print(f"{key} has \n{val}\n")


    return "returning kmeans"


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
