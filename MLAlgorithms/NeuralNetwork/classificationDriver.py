import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from MLAlgorithms.NeuralNetwork.NeuralNetwork import NeuralNetwork
from MLAlgorithms.Utils.KFolds import KFolds
from MLAlgorithms.Utils.StandardNormalizer import StandardNormalizer
from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.ClassifierAnalyzer import ClassifierAnalyzer
from MLAlgorithms.Utils.OneHotEncoder import OneHotEncoder

dataRetriever = DataRetriever("../Datasets/metadata.json")
dataRetriever.retrieveData("glass")
dataset = dataRetriever.getDataSet().dropna()
dataset = dataset.reset_index(drop=True)

# This line is used to normalize the data for Forest Fires
# dataset[dataRetriever.getDataClass()] = np.log(dataset[dataRetriever.getDataClass()]+0.1)
maxIter = 50
learning_rate = 0.01
batch_size = 0.2
for test_set, train_set in tqdm(KFolds(dataset, 10)):
    test_set = test_set.reset_index(drop=True)
    train_set = train_set.reset_index(drop=True)
    ohe = OneHotEncoder()
    discrete_attr = dataRetriever.getDescreteAttributes()
    if dataRetriever.getDataClass() in discrete_attr:
        discrete_attr.remove(dataRetriever.getDataClass())

    train_set = ohe.train_fit(train_set, discrete_attr)
    test_set = ohe.fit(test_set)

    sn = StandardNormalizer(train_set[dataRetriever.getContinuousAttributes()])
    train_set[dataRetriever.getContinuousAttributes()] = sn.train_fit()
    test_set[dataRetriever.getContinuousAttributes()] = sn.fit(test_set[dataRetriever.getContinuousAttributes()])

    nn = NeuralNetwork(train_set, 2, [3, 3], dataRetriever.getPredictionType(), dataRetriever.getDataClass())
    nn.train(maxIter, learning_rate, batch_size)

    predictions = nn.test(test_set.drop(dataRetriever.getDataClass(), axis=1))
