from MLAlgorithms.NaiveBayes.naiveBayes import NaiveBayes
from MLAlgorithms.Utils.BinDiscretizer import BinDiscretizer
from MLAlgorithms.Utils.ClassifierAnalyzer import ClassifierAnalyzer
from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.KFolds import KFolds
from MLAlgorithms.Utils.RangeNormalizer import RangeNormalizer
from MLAlgorithms.Utils.StandardNormalizer import StandardNormalizer
import pandas as pd
import math as m
import json


def calculateResults(predictions, answers):
    """Calculates the accuracy of the NaiveBayes algorithm

    Args:
        predictions (List<Predictions>): The result of the algorithm
        answers (List<Truths>): The answer to each row fed into the algorithm
    """
    t = 0
    f = 0
    for i in range(len(answers)):
        if predictions[i] == answers[i]:
            t += 1
            # print("SUCCESS: Prediction is {p1} and Answer is {a}".format(p1=predictions[i], a=answers[i]))
        else:
            f += 1
            # print("FAILURE: Prediction is {p1} and Answer is {a}".format(p1=predictions[i], a=answers[i]))

    print("The Percent True is {t}".format(t=t / len(answers)))
    print("The Percent False is {f}\n".format(f=f / len(answers)))


dataRetriever = DataRetriever("../Datasets/metadata.json")

jsonResults1 = {}
for dataSet in dataRetriever.getDataMenu():
    dataRetriever.retrieveData(dataSet)
    dataClass = dataRetriever.getDataClass()
    retrievedData = dataRetriever.getDataSet()

    numOfClassValues = len(retrievedData[dataRetriever.getDataClass()].unique())
    method = "macro"
    foldNum = 1

    jsonResults1[dataSet] = {}
    jsonResults1[dataSet]["isShuffled"] = False

    print(f"CURRENTLY PRINTING RESULTS FOR THE DATASET {dataSet}")
    for train, test in KFolds(retrievedData, 10):

        trainBin = BinDiscretizer(train[dataRetriever.getContinuousAttributes()], multi=True)

        trainBin.train_multi()
        train[dataRetriever.getContinuousAttributes()] = trainBin.fit_multi(
            train[dataRetriever.getContinuousAttributes()])
        test[dataRetriever.getContinuousAttributes()] = trainBin.fit_multi(
            test[dataRetriever.getContinuousAttributes()])

        naiveBayes = NaiveBayes(train, dataClass)

        answers = test[dataClass].to_numpy()[:]
        test = test.drop(columns=dataClass)
        predictions = naiveBayes.test(test)

        classifierAnalyzer = ClassifierAnalyzer(answers, predictions)


        # print(set(answers))
        # print(set(predictions))
        #
        # print(dataSet, foldNum)
        jsonResults1[dataSet][foldNum] = {}
        jsonResults1[dataSet][foldNum]["Precision"] = classifierAnalyzer.calc_precision(method=method)
        jsonResults1[dataSet][foldNum]["Recall"] = classifierAnalyzer.calc_recall(method=method)
        jsonResults1[dataSet][foldNum]["Accuracy"] = classifierAnalyzer.calc_accuracy(method=method)
        jsonResults1[dataSet][foldNum]["F1Score"] = classifierAnalyzer.calc_f1_score(method=method)
        foldNum += 1

        calculateResults(predictions, answers)

jsonResults2 = {}

for dataSet in dataRetriever.getDataMenu():
    dataRetriever.retrieveData(dataSet)
    dataClass = dataRetriever.getDataClass()
    retrievedData = dataRetriever.getDataSet()

    numOfRandFeatures = int(m.ceil(len(retrievedData.columns) * .1))
    onlyClasses = retrievedData[dataRetriever.getDataClass()]
    onlyData = retrievedData.drop([dataRetriever.getDataClass()], axis=1)

    dataToShuffle = onlyData.sample(numOfRandFeatures, axis=1, random_state=5)

    numOfClassValues = len(retrievedData[dataRetriever.getDataClass()].unique())
    method = "macro"
    foldNum = 1

    jsonResults2[dataSet] = {}
    jsonResults2[dataSet]["isShuffled"] = False


    for i, col in enumerate(dataToShuffle.columns):
        onlyData.drop([col], axis=1)
        onlyData[col] = dataToShuffle[col].sample(frac=1, random_state=i).to_numpy()

    onlyData[dataRetriever.getDataClass()] = onlyClasses


    print(f"CURRENTLY PRINTING RESULTS FOR THE SHUFFLED DATASET {dataSet}")
    for train, test in KFolds(onlyData, 10):
        jsonResults2[dataSet][foldNum] = {}
        trainBin = BinDiscretizer(train[dataRetriever.getContinuousAttributes()], multi=True)

        trainBin.train_multi()
        train[dataRetriever.getContinuousAttributes()] = trainBin.fit_multi(
            train[dataRetriever.getContinuousAttributes()])
        test[dataRetriever.getContinuousAttributes()] = trainBin.fit_multi(
            test[dataRetriever.getContinuousAttributes()])

        naiveBayes = NaiveBayes(train, dataClass)

        answers = test[dataClass].to_numpy()[:]
        test = test.drop(columns=dataClass)
        predictions = naiveBayes.test(test)


        classifierAnalyzer = ClassifierAnalyzer(answers, predictions)

        jsonResults2[dataSet][foldNum]["Precision"] = classifierAnalyzer.calc_precision(method=method)
        jsonResults2[dataSet][foldNum]["Recall"] = classifierAnalyzer.calc_recall(method=method)
        jsonResults2[dataSet][foldNum]["Accuracy"] = classifierAnalyzer.calc_accuracy(method=method)
        jsonResults2[dataSet][foldNum]["F1Score"] = classifierAnalyzer.calc_f1_score(method=method)
        foldNum += 1

        calculateResults(predictions, answers)


with open("Nonshuffled.json", 'w') as f:
    f.write(json.dumps(jsonResults1, indent=2))

with open("Shuffled.json", 'w') as f:
    f.write(json.dumps(jsonResults2, indent=2))