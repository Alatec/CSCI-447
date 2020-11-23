from MLAlgorithms.KNN.KNearest import KNearestNeighbor, EditedKNN, CondensedKNN
from MLAlgorithms.Utils.KFolds import KFolds
from MLAlgorithms.Utils.StandardNormalizer import StandardNormalizer
from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.ClassifierAnalyzer import ClassifierAnalyzer

import numpy as np
import pandas as pd
import json
import glob
import json

dataRetriever = DataRetriever("../Datasets/metadata.json")
dataRetriever.retrieveData("glass")
data = dataRetriever.getDataSet()
data = data.dropna()
data = data.sample(frac=1.0, random_state=93)
data = data.reset_index(drop=True)

# data = data.drop('region-pixel-count', axis=1)

class_col = dataRetriever.getDataClass()
# data[class_col] = np.log(data[class_col] + 0.001)


contAttr = dataRetriever.getContinuousAttributes()
# contAttr.remove('region-pixel-count')
discAttr = dataRetriever.getDescreteAttributes()
predictionType = dataRetriever.getPredictionType()
f = open("glassPerf.json",'r')
output_json = json.load(f)
f.close()
iter_num = 0
centroidsTrain = pd.read_csv("CSVOutput/normalizedglassKMeansClustered.csv")
medoidsTrain = pd.read_csv("CSVOutput/normalizedglassMedoidsClustered.csv")
for test, train in KFolds(data, 5, stratisfied=True, class_col=class_col):

    #KFolds doesn't have the capability of returning a validate set
    #K is set to desired k/2 and the validate set is half of the test set

    sn = StandardNormalizer(train[contAttr])
    train[contAttr] = sn.train_fit()

    test1 = test.sample(frac=0.5, random_state=13)
    test2 = test.drop(test1.index)

    train = train.reset_index(drop=True)
    test1  = test1.reset_index(drop=True)
    test2 = test2.reset_index(drop=True)

    test1[contAttr] = sn.fit(test1[contAttr])
    test2[contAttr] = sn.fit(test2[contAttr])

    k_vals = list(output_json[str(iter_num)].keys())
    k_vals_int = [int(k) for k in k_vals]
    k_vals_int[-1] = 18

    print(f"Fold {iter_num}")
    #Fold 1
    centroidsKNN = KNearestNeighbor(test1.drop(class_col, axis=1), centroidsTrain, k_vals_int, contAttr, discAttr, unknown_col=class_col, predictionType="classification")
    centPreds = centroidsKNN.test()

    medoidsKNN = KNearestNeighbor(test1.drop(class_col, axis=1), medoidsTrain, k_vals_int, contAttr, discAttr, unknown_col=class_col, predictionType="classification")
    medPreds =medoidsKNN.test()

    
    # output_json[iter_num] = {}
    for i, k in enumerate(k_vals):
        CentClassAnalyzer = ClassifierAnalyzer(test1[class_col], centPreds[:,i])
        MedClassAnalyzer = ClassifierAnalyzer(test1[class_col], medPreds[:,i])


        
        # output_json[iter_num][k] = {}

        #Centroids
        output_json[str(iter_num)][str(k)]["cent"] = {
            "F1": CentClassAnalyzer.calc_f1_score("macro"),
            "P": CentClassAnalyzer.calc_f1_score("macro"),
            "R": CentClassAnalyzer.calc_f1_score("macro"),
            "A": CentClassAnalyzer.calc_f1_score("macro"),
        }

        #Medoids
        output_json[str(iter_num)][str(k)]["med"] = {
            "F1": MedClassAnalyzer.calc_f1_score("macro"),
            "P": MedClassAnalyzer.calc_f1_score("macro"),
            "R": MedClassAnalyzer.calc_f1_score("macro"),
            "A": MedClassAnalyzer.calc_f1_score("macro"),
        }

    
    iter_num += 1

    print(f"Fold {iter_num}")
    #Fold 2
    centroidsKNN = KNearestNeighbor(test2.drop(class_col, axis=1), centroidsTrain, k_vals_int, contAttr, discAttr, unknown_col=class_col, predictionType="classification")
    centPreds = centroidsKNN.test()

    medoidsKNN = KNearestNeighbor(test2.drop(class_col, axis=1), medoidsTrain, k_vals_int, contAttr, discAttr, unknown_col=class_col, predictionType="classification")
    medPreds = medoidsKNN.test()

    
    # output_json[iter_num] = {}
    for i, k in enumerate(k_vals):
        CentClassAnalyzer = ClassifierAnalyzer(test2[class_col], centPreds[:,i])
        MedClassAnalyzer = ClassifierAnalyzer(test2[class_col], medPreds[:,i])


        
        # output_json[iter_num][k] = {}

        #Centroids
        output_json[str(iter_num)][str(k)]["cent"] = {
            "F1": CentClassAnalyzer.calc_f1_score("macro"),
            "P": CentClassAnalyzer.calc_f1_score("macro"),
            "R": CentClassAnalyzer.calc_f1_score("macro"),
            "A": CentClassAnalyzer.calc_f1_score("macro"),
        }

        #Medoids
        output_json[str(iter_num)][str(k)]["med"] = {
            "F1": MedClassAnalyzer.calc_f1_score("macro"),
            "P": MedClassAnalyzer.calc_f1_score("macro"),
            "R": MedClassAnalyzer.calc_f1_score("macro"),
            "A": MedClassAnalyzer.calc_f1_score("macro"),
        }

    
    iter_num += 1



with open("glassPerfWClust.json", 'w') as f:
    f.write(json.dumps(output_json, indent=2))