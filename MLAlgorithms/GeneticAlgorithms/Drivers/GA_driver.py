from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.NeuralNetwork.NeuralNetwork import NeuralNetwork
from MLAlgorithms.Utils.OneHotEncoder import OneHotEncoder
from MLAlgorithms.GeneticAlgorithms.differential_mutation import differential_mutation
from MLAlgorithms.GeneticAlgorithms.differential_mutation import differential_binomial_crossover

from MLAlgorithms.Utils.KFolds import KFolds
from MLAlgorithms.Utils.StandardNormalizer import StandardNormalizer



from tqdm import tqdm
import numpy as np
import pandas as pd
import calendar
# import matplotlib
# print(matplotlib.rcsetup.interactive_bk)
# matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import random as rand
import json

import ray

@ray.remote
def multiprocess_func(test_set, train_set, fold, fitness_file, output_file, dataRetriever, cost_func, current_data_set, mutation_rate, maxIter, batch_size, population_size, network_architecture, pb_actor=None):
    
    print("=========================")
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
    nn = NeuralNetwork(train_set, len(network_architecture), network_architecture, dataRetriever.getPredictionType(), dataRetriever.getDataClass())
    
    fitnesses = nn.genetic_algorithm(population_size, maxIter, batch_size, mutation_rate, 10, cost_func)
    final = nn.test(test_set.drop(dataRetriever.getDataClass(), axis=1))
    output = nn._feed_forward(test_set.drop(dataRetriever.getDataClass(), axis=1), testing=True)
    actual = test_set[dataRetriever.getDataClass()]

    # output_json[f"Fold {fold}"] = {}

    # fitness_file = f"../DataDump/{current_data_set}_fold{fold}_fitness.csv"
    fitness_pd = pd.DataFrame(fitnesses,columns=["Max_Weight", "Min_Weight", "Mean_Fitness"])
    fitness_pd.to_csv(fitness_file, index=False)

    # output_json[f"Fold {fold}"]["fitness"] = fitness_file
    

    # output_file = f"../DataDump/GA/{current_data_set}_fold{fold}_output.csv"
    
    
    # output_pd.to_csv(output_file, index=False)
    # output_json[f"Fold {fold}"]["results"] = output_file
    
    
    

    print("Fold Performance:")
    if dataRetriever.getPredictionType() == "classification":
    # ## ===================== Classification =================
        output_pd = pd.DataFrame({'Truth':actual.to_list(), 'Predicted':final})
        output_pd.to_csv(output_file, index=False)
        correct = 0
        for i, row in enumerate(final):
            if row == actual.iloc[i]: correct += 1

        acc = correct/len(test_set)

        # metrics.append(acc)
        print(f"Accuracy: {acc}")
        return acc
    else:
        output = output.reshape(output.shape[0])
        output_pd = pd.DataFrame({'Truth':actual.to_list(), 'Predicted':output})
        output_pd.to_csv(output_file, index=False)
        
        res = actual-output
        r2 = 1-((res**2).sum()/(((actual-actual.mean())**2).sum()))
        # metrics.append(r2)
        print(f"R2: {r2}")
        return float(r2)



def run_driver(current_data_set, mutation_rate=0.5, maxIter=1000, batch_size=0.6, population_size=110, network_architecture=[15], pb_actor=None):
    cost_func = {"breastCancer": "bin_cross", 
    "glass": "log_cosh", 
    "soybeanSmall": "log_cosh", 
    "abalone": "log_cosh", 
    "forestFires": "log_cosh",
    "computerHardware":"log_cosh"}

    title_text = r""" 
       ______                    __   _          ___     __                     _  __   __                    
      / ____/___   ____   ___   / /_ (_)_____   /   /   / /____ _ ____   _____ (_)/ /_ / /_   ____ ___   _____
     / / __ / _ \ / __ \ / _ \ / __// // ___/  / /| /  / // __ `// __ \ / ___// // __// __ \ / __ `__ \ / ___/
    / /_/ //  __// / / //  __// /_ / // /__   / ___ / / // /_/ // /_/ // /   / // /_ / / / // / / / / /(__  ) 
    \____/ \___//_/ /_/ \___/ \__//_/ \___/  /_/  |_//_/ \__, / \____//_/   /_/ \__//_/ /_//_/ /_/ /_//____/  
                                                        /____/                                                
    """

    output_json = {}

    # ====================== Adjustable Variables ==============================
    # current_data_set = "abalone"
    # mutation_rate = .5
    # maxIter = 10
    # batch_size = .6
    # population_size = 110

    # network_architecture = []
    # ===========================================================================

    output_json["parameters"] = {
        "mutation_rate": mutation_rate,
        "population_size": population_size,
        "network_architecture":network_architecture,
        "cost_func": cost_func[current_data_set],
        "maxIter":maxIter,
        "batch_size":batch_size
    }

    # ================ Data pre-processing =================================================
    dataRetriever = DataRetriever("../../Datasets/metadata.json")
    dataRetriever.retrieveData(current_data_set)
    dataset = dataRetriever.getDataSet().dropna()

    discrete_attr = dataRetriever.getDescreteAttributes()
    cont_attributes = dataRetriever.getContinuousAttributes()
    # This line is used to normalize the data for Forest Fires
    if current_data_set == "forestFires":
        # zeros = dataset[dataset[dataRetriever.getDataClass()] < 1].index
        # print(len(zeros)/len(dataset))
        # dataset = dataset.drop(zeros)
        discrete_attr.remove('month')
        discrete_attr.remove('day')
        # print(dataset[['month','day']])
        dataset['month'] = (pd.to_datetime(dataset.month, format='%b').dt.month) - 1
        dataset["day"] = dataset['day'].apply(lambda x: list(calendar.day_abbr).index(x.capitalize()))
        dataset["month_sin"] = np.sin(dataset['month'])
        dataset["month_cos"] = np.sin(dataset['month'])

        dataset["day_sin"] = np.sin(dataset['day'])
        dataset["day_cos"] = np.sin(dataset['day'])
        dataset = dataset.drop('day',axis=1)
        dataset = dataset.drop('month',axis=1)
        cont_attributes.append('month_sin')
        cont_attributes.append('month_cos')
        cont_attributes.append('day_sin')
        cont_attributes.append('day_cos')
        # print(dataset[['month','day']])
        
        dataset[dataRetriever.getDataClass()] = np.log(dataset[dataRetriever.getDataClass()]+0.000001)
    elif current_data_set == "computerHardware":
        discrete_attr.remove('venderName')
        discrete_attr.remove('modelName')
        dataset = dataset.drop('venderName',axis=1)
        dataset = dataset.drop('modelName',axis=1)


    dataset = dataset.reset_index(drop=True)


    if dataRetriever.getDataClass() in discrete_attr:
        discrete_attr.remove(dataRetriever.getDataClass())


    # ======================= Train Neural Network ================
    print(title_text)
    fold = 0
    metrics = []

    for test_set, train_set in KFolds(dataset, 10):
        fold += 1
        fitness_file = f"../DataDump/GA/{current_data_set}_layer{len(network_architecture)}_fold{fold}_fitness.csv"
        output_file = f"../DataDump/GA/{current_data_set}_layer{len(network_architecture)}_fold{fold}_output.csv"
        # output_json[f"Fold {fold}"] = {}
        # output_json[f"Fold {fold}"]["fitness"] = fitness_file
        # output_json[f"Fold {fold}"]["results"] = output_file

        metrics.append(multiprocess_func.remote(test_set, train_set, fold, fitness_file, output_file, dataRetriever, cost_func[current_data_set], current_data_set, mutation_rate, maxIter, batch_size, population_size, network_architecture, pb_actor=None))


    metrics = ray.get(metrics)
    print(metrics)
    print("Average Performance: ", np.asarray(metrics).mean())
    output_json["Metrics"] = metrics
    output_json["Average"] = np.asarray(metrics, dtype=np.float64).mean()
    output_json["Std"] = np.asarray(metrics, dtype=np.float64).std()

    with open(f"../DataDump/GA_{current_data_set}_layer{len(network_architecture)}.json", 'w') as f:
        json.dump(output_json,f, indent=4)