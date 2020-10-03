from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.KNN.KMeansClustering import KMeans
from MLAlgorithms.Utils.StandardNormalizer import StandardNormalizer

import numpy as np


data = DataRetriever("../Datasets/metadata.json")


dataSetName = "computerHardware"


print(f"Creating CSV for {dataSetName}")
data.retrieveData(dataSetName)

maxItter = 100
kValue = 78

dataSetUnNormalized = data.getDataSet()
dataSetUnNormalized[data.getDataClass()] = np.log(dataSetUnNormalized[data.getDataClass()] + 0.001)  #// This is for Forest Fires

sn = StandardNormalizer(dataSetUnNormalized[data.getContinuousAttributes()])
dataSetUnNormalized[data.getContinuousAttributes()] = sn.train_fit()


dataSetNormalized = dataSetUnNormalized

# dataSetNormalized = dataSetNormalized.drop(["idNumber"], axis=1) // For Glass

centroids = KMeans(dataSetNormalized, data.getDataClass(), data.getDescreteAttributes(),
                   data.getContinuousAttributes(), data.getPredictionType(), kValue, maxItter)

centroids.to_csv('./CSVOutput/' + "normalized" + dataSetName + 'KMeansClustered.csv', index=False)
print(f"CSV for " + dataSetName + " has been created!")
