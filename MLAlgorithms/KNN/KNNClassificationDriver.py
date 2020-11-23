from MLAlgorithms.KNN.KNearest import KNearestNeighbor, EditedKNN, CondensedKNN
from MLAlgorithms.Utils.KFolds import KFolds
from MLAlgorithms.Utils.StandardNormalizer import StandardNormalizer
from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.ClassifierAnalyzer import ClassifierAnalyzer

import numpy as np
import json



dataRetriever = DataRetriever("../Datasets/metadata.json")
dataRetriever.retrieveData("vote")
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

    k_vals = [1,3,5,7, int(np.floor(np.sqrt(len(train))))]

    print(f"Fold {iter_num}")
    #Fold 1
    KNN = KNearestNeighbor(test1.drop(class_col, axis=1), train, k_vals, contAttr, discAttr, unknown_col=class_col, predictionType="classification")
    KPreds = KNN.test()

    CKNN = CondensedKNN(test1.drop(class_col, axis=1), train, k_vals, contAttr, discAttr, unknown_col=class_col, predictionType="classification")
    CKNN.train()
    CPreds = CKNN.test()

    EKNN = EditedKNN(test1.drop(class_col, axis=1), test2, train, k_vals, contAttr, discAttr, unknown_col=class_col, val_unknown_col=class_col, predictionType="classification")
    EKNN.train()
    EPreds = CKNN.test()
    output_json[iter_num] = {}
    for i, k in enumerate(k_vals):
        KClassAnalyzer = ClassifierAnalyzer(test1[class_col], KPreds[:,i])
        CClassAnalyzer = ClassifierAnalyzer(test1[class_col], CPreds[:,i])
        EClassAnalyzer = ClassifierAnalyzer(test1[class_col], EPreds[:,i])

        
        output_json[iter_num][k] = {}

        #KNN
        output_json[iter_num][k]["KNN"] = {
            "F1": KClassAnalyzer.calc_f1_score("macro"),
            "P": KClassAnalyzer.calc_f1_score("macro"),
            "R": KClassAnalyzer.calc_f1_score("macro"),
            "A": KClassAnalyzer.calc_f1_score("macro"),
        }

        #CNN
        output_json[iter_num][k]["CNN"] = {
            "F1": CClassAnalyzer.calc_f1_score("macro"),
            "P": CClassAnalyzer.calc_f1_score("macro"),
            "R": CClassAnalyzer.calc_f1_score("macro"),
            "A": CClassAnalyzer.calc_f1_score("macro"),
        }

        #ENN
        output_json[iter_num][k]["ENN"] = {
            "F1": EClassAnalyzer.calc_f1_score("macro"),
            "P": EClassAnalyzer.calc_f1_score("macro"),
            "R": EClassAnalyzer.calc_f1_score("macro"),
            "A": EClassAnalyzer.calc_f1_score("macro"),
        }

    print(output_json)

    
    iter_num += 1

    print(f"Fold {iter_num}")
    #Fold 2
    KNN = KNearestNeighbor(test1.drop(class_col, axis=1), train, k_vals, contAttr, discAttr, unknown_col=class_col, predictionType="classification")
    KPreds = KNN.test()

    CKNN = CondensedKNN(test1.drop(class_col, axis=1), train, k_vals, contAttr, discAttr, unknown_col=class_col, predictionType="classification")
    CKNN.train()
    CPreds = CKNN.test()

    EKNN = EditedKNN(test1.drop(class_col, axis=1), test2, train, k_vals, contAttr, discAttr, unknown_col=class_col, val_unknown_col=class_col, predictionType="classification")
    EKNN.train()
    EPreds = CKNN.test()

    output_json[iter_num] = {}
    for i, k in enumerate(k_vals):
        KClassAnalyzer = ClassifierAnalyzer(test1[class_col], KPreds[:,i])
        CClassAnalyzer = ClassifierAnalyzer(test1[class_col], CPreds[:,i])
        EClassAnalyzer = ClassifierAnalyzer(test1[class_col], EPreds[:,i])

        
        output_json[iter_num][k] = {}

        #KNN
        output_json[iter_num][k]["KNN"] = {
            "F1": KClassAnalyzer.calc_f1_score("macro"),
            "P": KClassAnalyzer.calc_f1_score("macro"),
            "R": KClassAnalyzer.calc_f1_score("macro"),
            "A": KClassAnalyzer.calc_f1_score("macro"),
        }

        #CNN
        output_json[iter_num][k]["CNN"] = {
            "F1": CClassAnalyzer.calc_f1_score("macro"),
            "P": CClassAnalyzer.calc_f1_score("macro"),
            "R": CClassAnalyzer.calc_f1_score("macro"),
            "A": CClassAnalyzer.calc_f1_score("macro"),
        }

        #ENN
        output_json[iter_num][k]["ENN"] = {
            "F1": EClassAnalyzer.calc_f1_score("macro"),
            "P": EClassAnalyzer.calc_f1_score("macro"),
            "R": EClassAnalyzer.calc_f1_score("macro"),
            "A": EClassAnalyzer.calc_f1_score("macro"),
        }


    
    iter_num += 1
    break

with open("imageSegPerf.json", 'w') as f:
    f.write(json.dumps(output_json, indent=2))