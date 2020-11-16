from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.NeuralNetwork.NeuralNetwork import NeuralNetwork
from MLAlgorithms.Utils.OneHotEncoder import OneHotEncoder
from MLAlgorithms.GeneticAlgorithms.differential_mutation import differential_mutation
from MLAlgorithms.GeneticAlgorithms.differential_mutation import differential_binomial_crossover

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random as rand

cost_func = {"breastCancer": "bin_crosss", "glass": "bin_cross", "soybeanSmall": "bin_crosss", "abalone": "smoe", "forestFires": "amas"}

title_text = """ 
   ______                    __   _          ___     __                     _  __   __                    
  / ____/___   ____   ___   / /_ (_)_____   /   |   / /____ _ ____   _____ (_)/ /_ / /_   ____ ___   _____
 / / __ / _ \ / __ \ / _ \ / __// // ___/  / /| |  / // __ `// __ \ / ___// // __// __ \ / __ `__ \ / ___/
/ /_/ //  __// / / //  __// /_ / // /__   / ___ | / // /_/ // /_/ // /   / // /_ / / / // / / / / /(__  ) 
\____/ \___//_/ /_/ \___/ \__//_/ \___/  /_/  |_|/_/ \__, / \____//_/   /_/ \__//_/ /_//_/ /_/ /_//____/  
                                                    /____/                                                
"""


# ====================== Adjustable Variables ==============================
current_data_set = "abalone"
mutation_rate = .3
cross_over_prob = .7
learning_rate = 1e-6
maxItter = 100
batch_size = .1
population_size = 10
# ===========================================================================



# ================ Data pre-processing =================================================
dataRetriever = DataRetriever("../Datasets/metadata.json")
dataRetriever.retrieveData(current_data_set)
dataset = dataRetriever.getDataSet().dropna()

# # This line is used to normalize the data for Forest Fires
# dataset[dataRetriever.getDataClass()] = np.log(dataset[dataRetriever.getDataClass()]+0.1)

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


# ============== Create Neural Network ===========================
# NOTE: As of right now, the Neural Network only works for classification data sets
# nn = NeuralNetwork(datasetEncoded, 2,
#                    [2, 16], dataRetriever.getPredictionType(), dataRetriever.getDataClass())

# ================================================================


# ======================= Train Neural Network ================
print(title_text)
perf = []
actual = testEncoded[dataRetriever.getDataClass()]

# Initialize a population of individuals
#     -> This is a bunch of randomly generated weigh matrices
# While stopping condition not True
#     For each individual
#         Check the fitness
#             -> This is done via cost function
#         Create a trial vector via mutation
#         Create offspring via crossover
#         If the fitness of the offspring is better than the parent
#             add offspring
#         else
#             add the parent

def stopping_cond():
    return True

population = [NeuralNetwork(datasetEncoded, 2, [2, 3], dataRetriever.getPredictionType(), 
                        dataRetriever.getDataClass(), seed=i) for i in range(population_size)]


i = 0
prev_result = 0
seed_counter = 0
for i in tqdm(range(maxItter)):
    current_result = 0

    for i in range(len(population)):
        random_individual_index = rand.sample([x for x in range(population_size) if x != i], 3)
        # Check fitness
        parent_fitness = population[i].differential_evolution_fitness(batch_size=batch_size, cost_func=cost_func[current_data_set])
        # Create Trial
        trial_matrix = differential_mutation(population[i].weight_matrix, 
                                            population[random_individual_index[0]].weight_matrix, 
                                            population[random_individual_index[1]].weight_matrix, 
                                            population[random_individual_index[2]].weight_matrix,
                                            mutation_rate)

        # Create offspring
        offspring_matrix = differential_binomial_crossover(population[i].weight_matrix, trial_matrix, cross_over_prob, seed=seed_counter)
        seed_counter += 1
        child = NeuralNetwork(datasetEncoded, 2, [2, 3], dataRetriever.getPredictionType(), 
                        dataRetriever.getDataClass())
        child.weight_matrix = offspring_matrix
        child_fitness = child.differential_evolution_fitness(batch_size=batch_size, cost_func=cost_func[current_data_set])

        # exit()
        final_parent_fitness = parent_fitness.sum().sum()
        final_child_fitness = child_fitness.sum().sum()
        if final_child_fitness < final_parent_fitness:
            population[i] = child

        final_fitness = final_child_fitness if final_child_fitness < final_parent_fitness else final_parent_fitness
        current_result += final_fitness


    # print((abs(current_result - prev_result) / abs(current_result + prev_result)))
    if ((abs(current_result - prev_result) / abs(current_result + prev_result)) > .0000001):
        prev_result = current_result
    else: 
        break
    # prev_result = current_result if ((abs(current_result - prev_result) / abs(current_result + prev_result)) > .00001) else return 


best = population[0]
for individual in population:
    # print(individual)
    if individual.differential_evolution_fitness().sum() > best.differential_evolution_fitness().sum():
        print('CHANGED')
        best = individual




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

    print()
    print("Everyone")
    for j in range(len(population)):
        be = population[j]
        final = be.test(testEncoded.drop(dataRetriever.getDataClass(), axis=1))
        output = be._feed_forward(testEncoded.drop(dataRetriever.getDataClass(), axis=1), testing=True)

        actual = testEncoded[dataRetriever.getDataClass()]


        ## ===================== Classification =================
        correct = 0
        for i, row in enumerate(final):
            if row == actual.iloc[i]: correct += 1

        acc = correct/len(test_set)

        print(final)
        print(np.array(actual))
        print(acc)

else:
    # ===================== Regression =================
    fig, axs = plt.subplots(3)
    output = output.reshape(output.shape[0])
    # output = ((output - output.mean())/output.std())
    # actual = (actual - actual.mean())/actual.std()
    rmse =(actual-output)


    # plt.hist(rmse)
    axs[0].hist(actual, label="Actual", alpha=0.5)
    axs[1].hist(output, label="Predicted", alpha=0.5)
    # axs[1].hist(rmse)
    # axs[0].legend()
    axs[2].scatter(actual, output-actual)
    plt.show()
