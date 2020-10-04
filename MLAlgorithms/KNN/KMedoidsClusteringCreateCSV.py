from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.KNN.KMediods import KMediods
from MLAlgorithms.Utils.StandardNormalizer import StandardNormalizer
import math as m

import numpy as np


data = DataRetriever("../Datasets/metadata.json")


dataSetName = "computerHardware"

print(f"Creating CSV for {dataSetName}")
data.retrieveData(dataSetName)

maxItter = 100
kValue = 78
# frac = .25
# random_state = 69
# kValue = m.floor(frac * kValue)



dataSetUnNormalized = data.getDataSet()
# dataSetUnNormalized[data.getDataClass()] = np.log(dataSetUnNormalized[data.getDataClass()] + 0.001)  // This is for Forest Fires

sn = StandardNormalizer(dataSetUnNormalized[data.getContinuousAttributes()])
dataSetUnNormalized[data.getContinuousAttributes()] = sn.train_fit()


dataSetNormalized = dataSetUnNormalized

# dataSetNormalized = dataSetNormalized.sample(frac=frac, random_state=random_state)
# dataSetNormalized = dataSetNormalized.reset_index()


# dataSetNormalized = dataSetNormalized.drop(["idNumber"], axis=1) #// For Glass

medoids = KMediods(dataSetNormalized, data.getDataClass(), data.getDescreteAttributes(),
                   data.getContinuousAttributes(), data.getPredictionType(), kValue, maxItter)

medoids.to_csv('./CSVOutput/' + "normalized" + dataSetName + 'MedoidsClustered.csv', index=False)
print(f"CSV for " + dataSetName + " has been created!")
