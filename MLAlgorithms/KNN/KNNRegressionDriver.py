from MLAlgorithms.KNN.KNearest import KNearestNeighbor, EditedKNN, CondensedKNN
from MLAlgorithms.Utils.KFolds import KFolds
from MLAlgorithms.Utils.StandardNormalizer import StandardNormalizer
from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.StandardNormalizer import StandardNormalizer


import numpy as np
import pandas as pd
import json

from tqdm import tqdm
#Regression Sets: Forest Fire, Hardware, computerHardwaree

dataRetriever = DataRetriever("../Datasets/metadata.json")
dataRetriever.retrieveData("abalone")
data = dataRetriever.getDataSet()
data = data.dropna()
data = data.sample(frac=1.0, random_state=93)
data = data.reset_index(drop=True)
# data = data.drop('idNumber', axis=1)

class_col = dataRetriever.getDataClass()
# data[class_col] = np.log(data[class_col] + 0.001)

centroidsTrain = pd.read_csv("CSVOutput/normalizedabaloneKMeansClustered.csv")
medoidsTrain = pd.read_csv("CSVOutput/normalizedabaloneMedoidsClustered.csv")

contAttr = dataRetriever.getContinuousAttributes()
discAttr = dataRetriever.getDescreteAttributes()
predictionType = dataRetriever.getPredictionType()

output_json = {}
iter_num = 0



for test, train in tqdm(KFolds(data, 10), total=1):
    k_vals = [1,3,5,7, int(np.floor(np.sqrt(len(train))))]

    #Normalize data
    sn = StandardNormalizer(train[contAttr+[class_col]])
    train[contAttr + [class_col]] = sn.train_fit()
    test[contAttr+ [class_col]] = sn.fit(test[contAttr+ [class_col]])

    # print("KNN")
    KNN = KNearestNeighbor(test.drop(class_col, axis=1), train, k_vals, contAttr, discAttr, unknown_col=class_col, predictionType=predictionType)
    # print("Cent")
    centKNN = KNearestNeighbor(test.drop(class_col, axis=1), centroidsTrain, [1,3,5,7, 10], contAttr, discAttr, unknown_col=class_col, predictionType=predictionType)
    # print("Med")
    medKNN = KNearestNeighbor(test.drop(class_col, axis=1), medoidsTrain, [1,3,5,7, 10], contAttr, discAttr, unknown_col=class_col, predictionType=predictionType)

    KPreds = KNN.test()
    KRes = KPreds - test[class_col].to_numpy()[:,None]

    centPreds = centKNN.test()
    centRes = centPreds - test[class_col].to_numpy()[:,None]

    medPreds  =  medKNN.test()
    medRes = medPreds - test[class_col].to_numpy()[:,None]

    mean = train[class_col].mean() # k = len(train)
    uRes = mean - test[class_col]

    output_json[iter_num] = {}
    for i, k in enumerate(k_vals):
        output_json[iter_num][k] = {}

        output_json[iter_num][k]["KNN"] = {
            "RMSE" : np.sqrt((KRes[:,i]**2).sum()),
            "R2"   : 1 - ((KRes[:, i]**2).sum()/(uRes**2).sum())
        }

        output_json[iter_num][k]["cent"] = {
            "RMSE" : np.sqrt((centRes[:,i]**2).sum()),
            "R2"   : 1 - ((centRes[:, i]**2).sum()/(uRes**2).sum())
        }

        output_json[iter_num][k]["med"] = {
            "RMSE" : np.sqrt((medRes[:,i]**2).sum()),
            "R2"   : 1 - ((medRes[:, i]**2).sum()/(uRes**2).sum())
        }

    output_json[iter_num]["mean"] = {
            "RMSE" : np.sqrt((uRes**2).sum()),
            "R2"   : 1 - ((uRes**2).sum()/(uRes**2).sum())
        }
    iter_num += 1
    break

with open("PerformanceOutput/abaloneVideoPerfWClust.json", 'w') as f:
    f.write(json.dumps(output_json, indent=2))