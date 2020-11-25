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

cost_func = {"breastCancer": "bin_crosss", 
"glass": "log_cosh", 
"soybeanSmall": "log_cosh", 
"abalone": "log_cosh", 
"forestFires": "asd",
"computerHardware":"log_cosh"}

title_text = r""" 
   ______                    __   _          ___     __                     _  __   __                    
  / ____/___   ____   ___   / /_ (_)_____   /   /   / /____ _ ____   _____ (_)/ /_ / /_   ____ ___   _____
 / / __ / _ \ / __ \ / _ \ / __// // ___/  / /| /  / // __ `// __ \ / ___// // __// __ \ / __ `__ \ / ___/
/ /_/ //  __// / / //  __// /_ / // /__   / ___ / / // /_/ // /_/ // /   / // /_ / / / // / / / / /(__  ) 
\____/ \___//_/ /_/ \___/ \__//_/ \___/  /_/  |_//_/ \__, / \____//_/   /_/ \__//_/ /_//_/ /_/ /_//____/  
                                                    /____/                                                
"""


# ====================== Adjustable Variables ==============================
current_data_set = "soybeanSmall"
mutation_rate = .5
maxItter = 1000
batch_size = .6
population_size = 110
# ===========================================================================



# ================ Data pre-processing =================================================
dataRetriever = DataRetriever("../Datasets/metadata.json")
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

dataset[cont_attributes] = (dataset[cont_attributes]
                                                    - dataset[cont_attributes].min())/(dataset[cont_attributes].max()-dataset[cont_attributes].min())

test_set = dataset.sample(frac=0.1, random_state=69)
train_set = dataset.drop(test_set.index)
test_set = test_set.reset_index(drop=True)
train_set = train_set.reset_index(drop=True)

ohe = OneHotEncoder()

if dataRetriever.getDataClass() in discrete_attr:
    discrete_attr.remove(dataRetriever.getDataClass())

datasetEncoded = ohe.train_fit(
    train_set, dataRetriever.getDescreteAttributes())
testEncoded = ohe.fit(test_set)

# ======================= Create Best Individual ================
print(title_text)

best = NeuralNetwork(datasetEncoded, 1, [25], dataRetriever.getPredictionType(), 
                            dataRetriever.getDataClass())
fitnesses = best.genetic_algorithm(population_size, maxItter, batch_size, mutation_rate, 10, cost_func[current_data_set])


# ======================= Test Best Individual ================
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

    plt.plot(fitnesses[:,0], c='blue', label='max')
    plt.legend()
    plt.savefig("PlotDump/Plot2.png")

else:
    # ===================== Regression =================
    fig, axs = plt.subplots(4)
    output = output.reshape(output.shape[0])
    rmse =(actual-output)

    axs[0].set_title('Actual')
    axs[0].hist(actual, label="Actual")
    ylim = axs[0].get_ylim()
    axs[0].plot([actual.mean(),actual.mean()],ylim)
    axs[0].plot([np.median(actual),np.median(actual)],ylim)

    axs[0].set_xlim([0,1])

    axs[1].set_title('Predicted')
    axs[1].hist(output, label="Predicted")
    axs[1].set_xlim([0,1])
    
    res = actual-output
    r2 = 1-((res**2).sum()/(((actual-actual.mean())**2).sum()))
    print(f"R2: {r2}")
    axs[2].hist(res)
    axs[2].set_title('Residuals')

    axs[3].set_title('Fitness')
    axs[3].plot(fitnesses[:,0], c='blue', label='mean')
    axs[3].plot(fitnesses[:,1], c='green',label='median')
    fig.tight_layout()
    plt.savefig("PlotDump/Plot1.png")
