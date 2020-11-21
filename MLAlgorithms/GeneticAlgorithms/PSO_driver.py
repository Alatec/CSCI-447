import pandas as pd
import numpy as np
from scipy.stats import norm

import matplotlib
# print(matplotlib.rcsetup.interactive_bk)
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from tqdm import tqdm

import os

from MLAlgorithms.NeuralNetwork.NeuralNetwork import NeuralNetwork
from MLAlgorithms.Utils.KFolds import KFolds
from MLAlgorithms.Utils.StandardNormalizer import StandardNormalizer
from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.ClassifierAnalyzer import ClassifierAnalyzer
from MLAlgorithms.Utils.OneHotEncoder import OneHotEncoder

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


happiness = r"""
    ____             __  _      __        _____                                 ____        __  _           _             __  _           
   / __ \____ ______/ /_(_)____/ /__     / ___/      ______ __________ ___     / __ \____  / /_(_)___ ___  (_)___  ____ _/ /_(_)___  ____ 
  / /_/ / __ `/ ___/ __/ / ___/ / _ \    \__ \ | /| / / __ `/ ___/ __ `__ \   / / / / __ \/ __/ / __ `__ \/ /_  / / __ `/ __/ / __ \/ __ \
 / ____/ /_/ / /  / /_/ / /__/ /  __/   ___/ / |/ |/ / /_/ / /  / / / / / /  / /_/ / /_/ / /_/ / / / / / / / / /_/ /_/ / /_/ / /_/ / / / /
/_/    \__,_/_/   \__/_/\___/_/\___/   /____/|__/|__/\__,_/_/  /_/ /_/ /_/   \____/ .___/\__/_/_/ /_/ /_/_/ /___/\__,_/\__/_/\____/_/ /_/ 
                                                                                 /_/
"""

print(happiness)
dataRetriever = DataRetriever("../Datasets/metadata.json")
dataRetriever.retrieveData("abalone")
dataset = dataRetriever.getDataSet().dropna()
dataset = dataset.reset_index(drop=True)

test_set = dataset.sample(frac=0.1, random_state=69)
train_set = dataset.drop(test_set.index)
test_set = test_set.reset_index(drop=True)
train_set = train_set.reset_index(drop=True)

# This line is used to normalize the data for Forest Fires
# dataset[dataRetriever.getDataClass()] = np.log(dataset[dataRetriever.getDataClass()]+0.1)
maxIter = 1
learning_rate = 1e-3
batch_size = 0.01

metrics = []
fold = 0


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
nn = NeuralNetwork(train_set, 2, [6,16], dataRetriever.getPredictionType(), dataRetriever.getDataClass())
fitness_matrix, average_fitness = nn._particle_swarm_optimize(70, max_iter=500)


predictions = nn._feed_forward(test_set.drop(dataRetriever.getDataClass(), axis=1), testing=True)

# # ca = ClassifierAnalyzer(test_set[dataRetriever.getDataClass()], predictions)
# correct = 0
actual = test_set[dataRetriever.getDataClass()]
# thresh = np.mean(predictions)
# for i, row in enumerate(predictions):
#     guess = 4
#     if row >= thresh: guess = 2
#     if guess == actual.iloc[i]: correct += 1
# metrics.append(correct/len(actual))

    

metrics = np.asarray(metrics)
# prior = 1/dataset[dataRetriever.getDataClass()].nunique()
# sampling_sd = np.sqrt((prior*(1-prior))/(10))


# p_score = 1-norm.cdf(np.median(metrics),loc=prior,scale=sampling_sd)
fig, ax = plt.subplots(3)
ax[0].plot(fitness_matrix[:,0], label="1")
ax[0].plot(fitness_matrix[:,1], label="34")
ax[0].plot(fitness_matrix[:,2], label="68")
ax[0].plot(fitness_matrix[:,3], label="Best")
ax[0].legend()
print(f"Average Accuracy: {np.asarray(metrics).mean()} ± {metrics.std()}")
print("Final Fold:")
print("Predicted Output: ",)
print(predictions)
print("Actual Output: ")
print(actual.to_numpy())
predictions = predictions.flatten()
# ax[1].hist(predictions)
ax[1].hist((predictions-predictions.mean())/predictions.std(), alpha=0.5, label='Predicted', density=True)
ax[1].hist((actual-actual.mean())/actual.std(), label='Actual', density=True, alpha=0.5)
ax[1].legend()

ax[2].plot(average_fitness)
plt.savefig("PlotDump/Plot1.png")
