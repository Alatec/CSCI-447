from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.NeuralNetwork.NeuralNetwork import NeuralNetwork
from MLAlgorithms.Utils.OneHotEncoder import OneHotEncoder
from MLAlgorithms.GeneticAlgorithms.differential_mutation import differential_mutation
from MLAlgorithms.GeneticAlgorithms.differential_mutation import differential_binomial_crossover

from MLAlgorithms.Utils.KFolds import KFolds
from MLAlgorithms.Utils.StandardNormalizer import StandardNormalizer

import ray
from tqdm import tqdm
import numpy as np
import pandas as pd
import calendar

import matplotlib.pyplot as plt
import random as rand
import json





@ray.remote
def multiprocess_func(test_set, train_set, fold, fitness_file, output_file, dataRetriever, cost_func, current_data_set, cross_over_prob=0.7,mutation_rate=0.3, maxIter=1000, batch_size=0.6, population_size=110, network_architecture=[15], pb_actor=None):
    
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
    
    fitnesses = nn.differential_evolution(population_size, maxIter, batch_size, mutation_rate, cross_over_prob, cost_func)
    final = nn.test(test_set.drop(dataRetriever.getDataClass(), axis=1))
    output = nn._feed_forward(test_set.drop(dataRetriever.getDataClass(), axis=1), testing=True)
    actual = test_set[dataRetriever.getDataClass()]

    fitness_pd = pd.DataFrame(fitnesses,columns=["Max_Weight", "Min_Weight", "Mean_Fitness"])
    fitness_pd.to_csv(fitness_file, index=False)

    print("Fold Performance:")
    if dataRetriever.getPredictionType() == "classification":
    # ## ===================== Classification =================
        correct = 0
        for i, row in enumerate(final):
            if row == actual.iloc[i]: correct += 1

        acc = correct/len(test_set)

        print(f"Accuracy: {acc}")
        output_pd = pd.DataFrame({'Truth':actual.to_list(), 'Predicted':final})
    
        output_pd.to_csv(output_file, index=False)
        return acc
    else:
        output = output.reshape(output.shape[0])
        
        res = actual-output
        r2 = 1-((res**2).sum()/(((actual-actual.mean())**2).sum()))
        print(f"R2: {r2}")
        output_pd = pd.DataFrame({'Truth':actual.to_list(), 'Predicted':output})
    
        output_pd.to_csv(output_file, index=False)
        return float(r2)


def run_driver(current_data_set, mutation_rate=0.3, cross_over_prob=0.7, maxIter=1000, batch_size=0.6, population_size=110, network_architecture=[15], pb_actor=None):
    cost_func = {"breastCancer": "bin_cross", 
    "glass": "log_cosh", 
    "soybeanSmall": "log_cosh", 
    "abalone": "hubers", 
    "forestFires": "log_cosh",
    "computerHardware": "log_cosh"}

    title_text = r"""
        ____  _ ________                     __  _       __   ______            __      __  _           
       / __ \(_) __/ __/__  ________  ____  / /_(_)___ _/ /  / ____/   ______  / /_  __/ /_(_)___  ____ 
      / / / / / /_/ /_/ _ \/ ___/ _ \/ __ \/ __/ / __ `/ /  / __/ | | / / __ \/ / / / / __/ / __ \/ __ \
     / /_/ / / __/ __/  __/ /  /  __/ / / / /_/ / /_/ / /  / /___ | |/ / /_/ / / /_/ / /_/ / /_/ / / / /
    /_____/_/_/ /_/  \___/_/   \___/_/ /_/\__/_/\__,_/_/  /_____/ |___/\____/_/\__,_/\__/_/\____/_/ /_/
    """

    output_json = {}

    # ====================== Adjustable Variables ==============================
    # current_data_set = "soybeanSmall"
    # mutation_rate = .3
    # cross_over_prob = .7
    # maxIter = 1000
    # batch_size = .1
    # population_size = 100
    # network_architecture = [15]
    # ===========================================================================

    output_json["parameters"] = {
        "mutation_rate": mutation_rate,
        "cross_over_prob": cross_over_prob,
        "population_size": population_size,
        "cost_function": cost_func[current_data_set],
        "maxIter": maxIter,
        "architecture": network_architecture
    }

    # ================ Data pre-processing =================================================
    dataRetriever = DataRetriever("../../Datasets/metadata.json")
    dataRetriever.retrieveData(current_data_set)
    dataset = dataRetriever.getDataSet().dropna()

    discrete_attr = dataRetriever.getDescreteAttributes()
    cont_attributes = dataRetriever.getContinuousAttributes()
    # This line is used to normalize the data for Forest Fires
    if current_data_set == "forestFires":
        discrete_attr.remove('month')
        discrete_attr.remove('day')
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
        fitness_file = f"../DataDump/DE/{current_data_set}_layer{len(network_architecture)}_fold{fold}_fitness.csv"
        output_file = f"../DataDump/DE/{current_data_set}_layer{len(network_architecture)}_fold{fold}_output.csv"

        metrics.append(multiprocess_func.remote(test_set, train_set, fold, fitness_file, output_file, dataRetriever, cost_func[current_data_set], 
            current_data_set, cross_over_prob=cross_over_prob,mutation_rate=mutation_rate, maxIter=maxIter, batch_size=batch_size, population_size=population_size, 
            network_architecture=network_architecture, pb_actor=None))


    metrics = ray.get(metrics)
    print(metrics)
    print("Average Performance: ", np.asarray(metrics).mean())
    output_json["Metrics"] = metrics
    output_json["Average"] = np.asarray(metrics, dtype=np.float64).mean()
    output_json["Std"] = np.asarray(metrics, dtype=np.float64).std()




    with open(f"../DataDump/DE_{current_data_set}_layer{len(network_architecture)}.json", 'w') as f:
        json.dump(output_json,f, indent=4)