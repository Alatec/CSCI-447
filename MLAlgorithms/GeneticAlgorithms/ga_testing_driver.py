from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.NeuralNetwork.NeuralNetwork import NeuralNetwork
from MLAlgorithms.Utils.OneHotEncoder import OneHotEncoder
from MLAlgorithms.GeneticAlgorithms.differential_mutation import differential_mutation
from MLAlgorithms.GeneticAlgorithms.differential_mutation import differential_binomial_crossover

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random as rand

cost_func = {"breastCancer": "bin_crosss", 
"glass": "bin_cross", 
"soybeanSmall": "bin_crosss", 
"abalone": "hubers", 
"forestFires": "amas",}

title_text = """ 
   ______                    __   _          ___     __                     _  __   __                    
  / ____/___   ____   ___   / /_ (_)_____   /   |   / /____ _ ____   _____ (_)/ /_ / /_   ____ ___   _____
 / / __ / _ \ / __ \ / _ \ / __// // ___/  / /| |  / // __ `// __ \ / ___// // __// __ \ / __ `__ \ / ___/
/ /_/ //  __// / / //  __// /_ / // /__   / ___ | / // /_/ // /_/ // /   / // /_ / / / // / / / / /(__  ) 
\____/ \___//_/ /_/ \___/ \__//_/ \___/  /_/  |_|/_/ \__, / \____//_/   /_/ \__//_/ /_//_/ /_/ /_//____/  
                                                    /____/                                                
"""


# ====================== Adjustable Variables ==============================
current_data_set = "forestFires"
mutation_rate = .1
maxItter = 100
batch_size = .4
population_size = 10
# ===========================================================================



# ================ Data pre-processing =================================================
dataRetriever = DataRetriever("../Datasets/metadata.json")
dataRetriever.retrieveData(current_data_set)
dataset = dataRetriever.getDataSet().dropna()

# This line is used to normalize the data for Forest Fires
if current_data_set == "forestFires":
    dataset[dataRetriever.getDataClass()] = np.log(dataset[dataRetriever.getDataClass()]+0.1)

dataset = dataset.reset_index(drop=True)
# dataset[dataRetriever.getContinuousAttributes()] = (dataset[dataRetriever.getContinuousAttributes()]
#                                                     - dataset[dataRetriever.getContinuousAttributes()].mean())/dataset[dataRetriever.getContinuousAttributes()].std()

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



best = NeuralNetwork(datasetEncoded, 2, [2, 3], dataRetriever.getPredictionType(), 
                            dataRetriever.getDataClass())
fitnesses = best.genetic_algorithm(population_size, maxItter, batch_size, mutation_rate, 10, cost_func[current_data_set])


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

else:
    # ===================== Regression =================
    fig, axs = plt.subplots(4)
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
    axs[3].plot(fitnesses)
    plt.show()
