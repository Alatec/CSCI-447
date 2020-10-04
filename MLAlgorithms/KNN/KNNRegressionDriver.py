from MLAlgorithms.KNN.KNearest import KNearestNeighbor, EditedKNN, CondensedKNN
from MLAlgorithms.Utils.KFolds import KFolds
from MLAlgorithms.Utils.StandardNormalizer import StandardNormalizer
from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.ClassifierAnalyzer import ClassifierAnalyzer

import numpy as np
import json



dataRetriever = DataRetriever("../Datasets/metadata.json")
dataRetriever.retrieveData("glass")
data = dataRetriever.getDataSet()
data = data.dropna()
data = data.sample(frac=1.0, random_state=93)
data = data.reset_index(drop=True)
# data = data.drop('idNumber', axis=1)

class_col = dataRetriever.getDataClass()
# data[class_col] = np.log(data[class_col] + 0.001)


contAttr = dataRetriever.getContinuousAttributes()
discAttr = dataRetriever.getDescreteAttributes()
predictionType = dataRetriever.getPredictionType()

output_json = {}
iter_num = 0