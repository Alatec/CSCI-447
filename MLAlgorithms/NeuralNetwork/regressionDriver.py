import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm

from MLAlgorithms.NeuralNetwork.NeuralNetwork import NeuralNetwork
from MLAlgorithms.Utils.KFolds import KFolds
from MLAlgorithms.Utils.StandardNormalizer import StandardNormalizer
from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.ClassifierAnalyzer import ClassifierAnalyzer
from MLAlgorithms.Utils.OneHotEncoder import OneHotEncoder

dataRetriever = DataRetriever("../Datasets/metadata.json")
dataRetriever.retrieveData("abalone")
dataset = dataRetriever.getDataSet().dropna()
dataset = dataset.reset_index(drop=True)

# This line is used to normalize the data for Forest Fires
# dataset[dataRetriever.getDataClass()] = np.log(dataset[dataRetriever.getDataClass()]+0.1)
maxIter = 1000
learning_rate = 1e-3
batch_size = 0.2

metrics = []
fold = 0

# Ten-Fold Cross Validation
for test_set, train_set in KFolds(dataset, 10):
    fold += 1
    print("Fold Num: ", fold)
    # Encode Data
    test_set = test_set.reset_index(drop=True)
    train_set = train_set.reset_index(drop=True)
    ohe = OneHotEncoder()
    discrete_attr = dataRetriever.getDescreteAttributes()
    if dataRetriever.getDataClass() in discrete_attr:
        discrete_attr.remove(dataRetriever.getDataClass())

    train_set = ohe.train_fit(train_set, discrete_attr)
    test_set = ohe.fit(test_set)

    #  Normalize Data
    sn = StandardNormalizer(train_set[dataRetriever.getContinuousAttributes()])
    train_set[dataRetriever.getContinuousAttributes()] = sn.train_fit()
    test_set[dataRetriever.getContinuousAttributes()] = sn.fit(test_set[dataRetriever.getContinuousAttributes()])

    # Train network and change architecture in respect to data set
    nn = NeuralNetwork(train_set, 2, [4,7], dataRetriever.getPredictionType(), dataRetriever.getDataClass())
    nn.train(maxIter, learning_rate, batch_size)

    predictions = nn.test(test_set.drop(dataRetriever.getDataClass(), axis=1))
    predictions = predictions.reshape(predictions.shape[0])
    actual = test_set[dataRetriever.getDataClass()]
    res = predictions-actual
    tot_ss = ((actual-actual.mean())**2).sum()
    res_ss = ((actual-predictions)**2).sum()
    R2 = 1-(res_ss/tot_ss)
    
    metrics.append(R2)

    

metrics = np.asarray(metrics)




print(f"Average R2: {np.asarray(metrics).mean()} ± me")
