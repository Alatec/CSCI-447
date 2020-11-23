from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.NeuralNetwork.NeuralNetwork import NeuralNetwork
from MLAlgorithms.Utils.OneHotEncoder import OneHotEncoder
from MLAlgorithms.GeneticAlgorithms.differential_mutation import differential_mutation
from MLAlgorithms.GeneticAlgorithms.differential_mutation import differential_binomial_crossover

from MLAlgorithms.Utils.KFolds import KFolds
from MLAlgorithms.Utils.StandardNormalizer import StandardNormalizer

import ray
# ray.init()

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

@ray.remote
def multiprocess_func(test_set, train_set, fold, fitness_file, output_file, dataRetriever, cost_func, current_data_set, mutation_rate=0.5, maxIter=1000, batch_size=0.6, population_size=110, network_architecture=[15], pb_actor=None):
    
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
    
    fitnesses = nn._particle_swarm_optimize(population_size, max_iter=maxIter, cost_func=cost_func,  pb_actor=pb_actor)
    final = nn.test(test_set.drop(dataRetriever.getDataClass(), axis=1))
    output = nn._feed_forward(test_set.drop(dataRetriever.getDataClass(), axis=1), testing=True)
    actual = test_set[dataRetriever.getDataClass()]

    # output_json[f"Fold {fold}"] = {}

    # fitness_file = f"../DataDump/{current_data_set}_fold{fold}_fitness.csv"
    fitness_pd = pd.DataFrame(fitnesses,columns=["Max_Weight", "Min_Weight", "Mean_Fitness"])
    fitness_pd.to_csv(fitness_file, index=False)

    # output_json[f"Fold {fold}"]["fitness"] = fitness_file
    

    # output_file = f"../DataDump/{current_data_set}_fold{fold}_output.csv"
    
    # output_json[f"Fold {fold}"]["results"] = output_file
    
    
    

    print("Fold Performance:")
    if dataRetriever.getPredictionType() == "classification":
    # ## ===================== Classification =================
        correct = 0
        for i, row in enumerate(final):
            if row == actual.iloc[i]: correct += 1

        acc = correct/len(test_set)

        # metrics.append(acc)
        print(f"Accuracy: {acc}")
        output_pd = pd.DataFrame({'Truth':actual.to_list(), 'Predicted':final})
    
        output_pd.to_csv(output_file, index=False)
        return acc
    else:
        output = output.reshape(output.shape[0])
        
        res = actual-output
        r2 = 1-((res**2).sum()/(((actual-actual.mean())**2).sum()))
        # metrics.append(r2)
        print(f"R2: {r2}")
        output_pd = pd.DataFrame({'Truth':actual.to_list(), 'Predicted':final})
    
        output_pd.to_csv(output_file, index=False)
        return float(r2)


def run_driver(current_data_set, mutation_rate=0.5, maxIter=1000, batch_size=0.6, population_size=110, network_architecture=[15], pb_actor=None):
    cost_func = {"breastCancer": "bin_cross", 
    "glass": "log_cosh", 
    "soybeanSmall": "log_cosh", 
    "abalone": "log_cosh", 
    "forestFires": "log_cosh",
    "computerHardware":"log_cosh"}

    title_text = r"""
        ____             __  _      __        _____                                 ____        __  _           _             __  _           
       / __ \____ ______/ /_(_)____/ /__     / ___/      ______ __________ ___     / __ \____  / /_(_)___ ___  (_)___  ____ _/ /_(_)___  ____ 
      / /_/ / __ `/ ___/ __/ / ___/ / _ \    \__ \ | /| / / __ `/ ___/ __ `__ \   / / / / __ \/ __/ / __ `__ \/ /_  / / __ `/ __/ / __ \/ __ \
     / ____/ /_/ / /  / /_/ / /__/ /  __/   ___/ / |/ |/ / /_/ / /  / / / / / /  / /_/ / /_/ / /_/ / / / / / / / / /_/ /_/ / /_/ / /_/ / / / /
    /_/    \__,_/_/   \__/_/\___/_/\___/   /____/|__/|__/\__,_/_/  /_/ /_/ /_/   \____/ .___/\__/_/_/ /_/ /_/_/ /___/\__,_/\__/_/\____/_/ /_/ 
                                                                                    /_/
    """

    output_json = {}

    # ====================== Adjustable Variables ==============================
    # current_data_set = "glass"
    # mutation_rate = .5
    # maxIter = 1000
    # batch_size = .6
    # population_size = 110

    # network_architecture = [15]
    # ===========================================================================

    output_json["parameters"] = {
        "mutation_rate": mutation_rate,
        "population_size": population_size,
        "cost_func": cost_func[current_data_set]
    }

    # ================ Data pre-processing =================================================
    dataRetriever = DataRetriever("../../Datasets/metadata.json")
    dataRetriever.retrieveData(current_data_set)
    dataset = dataRetriever.getDataSet().dropna()

    discrete_attr = dataRetriever.getDescreteAttributes()
    cont_attributes = dataRetriever.getContinuousAttributes()
    # This line is used to normalize the data for Forest Fires
    if current_data_set == "forestFires":
        zeros = dataset[dataset[dataRetriever.getDataClass()] < 1].index
        print(len(zeros)/len(dataset))
        dataset = dataset.drop(zeros)
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
        fitness_file = f"../DataDump/PSO/{current_data_set}_layer{len(network_architecture)}_fold{fold}_fitness.csv"
        output_file = f"../DataDump/PSO/{current_data_set}_layer{len(network_architecture)}_fold{fold}_output.csv"
        # output_json[f"Fold {fold}"] = {}
        # output_json[f"Fold {fold}"]["fitness"] = fitness_file
        # output_json[f"Fold {fold}"]["results"] = output_file

        metrics.append(multiprocess_func.remote(test_set, train_set, fold, fitness_file, output_file, dataRetriever, cost_func[current_data_set], current_data_set, mutation_rate=0.5, maxIter=1000, batch_size=0.6, population_size=110, network_architecture=[15], pb_actor=None))


    metrics = ray.get(metrics)
    print(metrics)
    print("Average Performance: ", np.asarray(metrics).mean())
    output_json["Metrics"] = metrics

    with open(f"../DataDump/PSO_{current_data_set}_layer{len(network_architecture)}.json", 'w') as f:
        json.dump(output_json,f, indent=4)