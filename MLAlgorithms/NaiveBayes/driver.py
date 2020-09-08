from MLAlgorithms.NaiveBayes.naiveBayes import NaiveBayes
from MLAlgorithms.Utils.BinDiscretizer import BinDiscretizer
from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.KFolds import KFolds
from MLAlgorithms.Utils.RangeNormalizer import RangeNormalizer

import pandas as pd
import numpy as np

# For train. test in KFOLD:
#   binData()
#   NormalizeData
#   Naivebayes

dataRetriever = DataRetriever("../Datasets/metadata.json")


dataRetriever.retrieveData("breastCancer")


for dataSet in dataRetriever.getDataMenu():
    dataRetriever.retrieveData(dataSet)
    dataClass = dataRetriever.getDataClass()
    print(f"CURRENTLY PRINTING RESULTS FOR THE DATASET {dataSet}")

    for train, test in KFolds(dataRetriever.getDataSet(), 10):
        bin = BinDiscretizer(train)
        normalizer = RangeNormalizer(train)


        naiveBayes = NaiveBayes(train, dataClass)

        answers = test[dataClass].to_numpy()
        test = test.drop(columns=dataClass)
        predictions = naiveBayes.test(test)

        t = 0
        f = 0
        for i in range(len(answers)):
            if predictions[i] == answers[i]:
                t += 1
                #print("SUCCESS: Prediction is {p1} and Answer is {a}".format(p1=predictions[i], a=answers[i]))
            else:
                f += 1
                #print("FAILURE: Prediction is {p1} and Answer is {a}".format(p1=predictions[i], a=answers[i]))

        print("The Percent True is {t}".format(t=t / len(answers)))
        print("The Percent False is {f}\n".format(f=f / len(answers)))


