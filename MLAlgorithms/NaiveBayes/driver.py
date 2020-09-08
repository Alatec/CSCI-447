from MLAlgorithms.NaiveBayes.naiveBayes import NaiveBayes
from MLAlgorithms.Utils.DataRetriever import DataRetriever

import pandas as pd
import numpy as np

# For train. test in KFOLD:
#   binData()
#   NormalizeData
#   Naivebayes

dataRetriever = DataRetriever("../Datasets/metadata.json")
dataRetriever.retrieveData("breastCancer")


naiveBayes = NaiveBayes(dataRetriever.getDataSet(), dataRetriever.getDataClass())

test = dataRetriever.getDataSet()
answers = test["class"].to_numpy()

test = test.drop(columns="class")
print(answers)
predictions = naiveBayes.test(test)
print(predictions)
t = 0
f = 0

for i in range(len(answers)):
    if predictions[i] == answers[i]:
        t += 1
        print("SUCCESS: Prediction is {p1} and Answer is {a}".format(p1=predictions[i], a=answers[i]))
    else:
        f += 1
        print("FAILURE: Prediction is {p1} and Answer is {a}".format(p1=predictions[i], a=answers[i]))

print(t/len(answers))
print(f/len(answers))