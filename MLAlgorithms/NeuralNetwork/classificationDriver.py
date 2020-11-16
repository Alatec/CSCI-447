import pandas as pd
import numpy as np

from scipy.stats import norm # Used for P score
import matplotlib.pyplot as plt

from tqdm import tqdm

from MLAlgorithms.NeuralNetwork.NeuralNetwork import NeuralNetwork
from MLAlgorithms.Utils.KFolds import KFolds
from MLAlgorithms.Utils.StandardNormalizer import StandardNormalizer
from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.ClassifierAnalyzer import ClassifierAnalyzer
from MLAlgorithms.Utils.OneHotEncoder import OneHotEncoder



dataRetriever = DataRetriever("../Datasets/metadata.json")
dataRetriever.retrieveData("breastCancer")
dataset = dataRetriever.getDataSet().dropna()
dataset = dataset.reset_index(drop=True)

# This line is used to normalize the data for Forest Fires
# dataset[dataRetriever.getDataClass()] = np.log(dataset[dataRetriever.getDataClass()]+0.1)
maxIter = 1
learning_rate = 1e-3
batch_size = 0.01

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
    nn = NeuralNetwork(train_set, 2, [2,2], dataRetriever.getPredictionType(), dataRetriever.getDataClass())
    nn.train(maxIter, learning_rate, batch_size)

    # predictions = nn.test(test_set.drop(dataRetriever.getDataClass(), axis=1))

    # # ca = ClassifierAnalyzer(test_set[dataRetriever.getDataClass()], predictions)
    # correct = 0
    # actual = test_set[dataRetriever.getDataClass()]
    # for i, row in enumerate(predictions):
    #     if row == actual.iloc[i]: correct += 1
    # metrics.append(correct/len(actual))
    break
    

metrics = np.asarray(metrics)
prior = 1/dataset[dataRetriever.getDataClass()].nunique()
sampling_sd = np.sqrt((prior*(1-prior))/(10))


# p_score = 1-norm.cdf(np.median(metrics),loc=prior,scale=sampling_sd)

# print(f"Average Accuracy: {np.asarray(metrics).mean()} ± {metrics.std()}")
# print("Final Fold:")
# print("Predicted Output: ",)
# print(predictions)
# print("Actual Output: ")
# print(actual.to_numpy())



