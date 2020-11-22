from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.NeuralNetwork.NeuralNetwork import NeuralNetwork
from MLAlgorithms.Utils.OneHotEncoder import OneHotEncoder
from MLAlgorithms.GeneticAlgorithms.differential_mutation import differential_mutation
from MLAlgorithms.GeneticAlgorithms.differential_mutation import differential_binomial_crossover

from tqdm import tqdm
import numpy as np
import pandas as pd
import calendar
import matplotlib.pyplot as plt
import random as rand

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


# ====================== Adjustable Variables ==============================
current_data_set = "glass"
mutation_rate = .3
cross_over_prob = .7
maxItter = 1000
batch_size = .1
population_size = 100
nodes_per_layer = [5, 9]
# ===========================================================================



# ================ Data pre-processing =================================================
dataRetriever = DataRetriever("../Datasets/metadata.json")
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
dataset[dataRetriever.getContinuousAttributes()] = (dataset[dataRetriever.getContinuousAttributes()]
                                                    - dataset[dataRetriever.getContinuousAttributes()].mean())/dataset[dataRetriever.getContinuousAttributes()].std()

test_set = dataset.sample(frac=0.1, random_state=69)
train_set = dataset.drop(test_set.index)
test_set = test_set.reset_index(drop=True)
train_set = train_set.reset_index(drop=True)

ohe = OneHotEncoder()
discrete_attr = dataRetriever.getDescreteAttributes()

if dataRetriever.getDataClass() in discrete_attr:
    discrete_attr.remove(dataRetriever.getDataClass())

datasetEncoded = ohe.train_fit(
    train_set, dataRetriever.getDescreteAttributes())
testEncoded = ohe.fit(test_set)



# ======================= Train Neural Network ================
print(title_text)


best = NeuralNetwork(datasetEncoded, len(nodes_per_layer), nodes_per_layer, dataRetriever.getPredictionType(), 
                            dataRetriever.getDataClass())
fitnesses = best.differential_evolution(population_size, maxItter, batch_size, mutation_rate, cross_over_prob, cost_func[current_data_set])


# print("Best")
final = best.test(testEncoded.drop(dataRetriever.getDataClass(), axis=1))
output = best._feed_forward(testEncoded.drop(dataRetriever.getDataClass(), axis=1), testing=True)
actual = testEncoded[dataRetriever.getDataClass()]
if dataRetriever.getPredictionType() == "classification":
    # ## ===================== Classification =================
    print("Best")
    correct = 0
    for i, row in enumerate(final):
        if row == actual.iloc[i]: correct += 1

    acc = correct/len(test_set)

    print(final)
    print(np.array(actual))
    print(acc)

    # print()
    # print("Everyone")
    # for j in range(len(population)):
    #     be = population[j]
    #     final = be.test(testEncoded.drop(dataRetriever.getDataClass(), axis=1))
    #     output = be._feed_forward(testEncoded.drop(dataRetriever.getDataClass(), axis=1), testing=True)

    #     actual = testEncoded[dataRetriever.getDataClass()]


    #     ## ===================== Classification =================
    #     correct = 0
    #     for i, row in enumerate(final):
    #         if row == actual.iloc[i]: correct += 1

    #     acc = correct/len(test_set)

    #     print(final)
    #     print(np.array(actual))
    #     print(acc)

        # plt.plot(fitnesses[:,0], c='blue', label='max')
    # plt.plot(fitnesses[:,0], c='green', label='max')
    plt.plot(fitnesses[:,1], c='blue', label='fitness')
    # plt.plot(fitnesses[:,2], c='orange', label='std')
    # plt.plot(fitnesses[:,3], c='red', label='min')
    # plt.plot(fitnesses[:,1]+1.5*fitnesses[:,2], c='black', label='outlier')
    # plt.plot(fitnesses[:,1]-1.5*fitnesses[:,2], c='black', label='outlier')
    # plt.yscale('log')
    plt.legend()
    # plt.plot(fitnesses[:,0]-fitnesses[:,1], c='green')
    plt.title(current_data_set)
    plt.show()

else:
    # ===================== Regression =================
    fig, axs = plt.subplots(3)
    output = output.reshape(output.shape[0])
    # output = ((output - output.mean())/output.std())
    # actual = (actual - actual.mean())/actual.std()
    rmse =(actual-output)


    # plt.hist(rmse)
    axs[0].hist(actual, label="Actual", alpha=0.5)
    axs[0].set_xlim([-2.5,5.5])
    axs[1].hist(output, label="Predicted", alpha=0.5)
    axs[1].set_xlim([-2.5,5.5])
    
    # axs[1].hist(rmse)
    # axs[0].legend()
    res = actual-output
    r2 = 1-((res**2).sum()/(((actual-actual.mean())**2).sum()))
    print(f"R2: {r2}")
    axs[2].hist(res)
    # axs[3].plot(fitnesses)
    plt.show()
