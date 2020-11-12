from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.NeuralNetwork.NeuralNetwork import NeuralNetwork
from MLAlgorithms.Utils.OneHotEncoder import OneHotEncoder
from MLAlgorithms.GeneticAlgorithms.differential_mutation import differential_mutation
from MLAlgorithms.GeneticAlgorithms.differential_mutation import differential_binomial_crossover

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random as rand

cost_func = {"breastCancer": "bin_cross"}

title_text = """ 
   ______                    __   _          ___     __                     _  __   __                    
  / ____/___   ____   ___   / /_ (_)_____   /   |   / /____ _ ____   _____ (_)/ /_ / /_   ____ ___   _____
 / / __ / _ \ / __ \ / _ \ / __// // ___/  / /| |  / // __ `// __ \ / ___// // __// __ \ / __ `__ \ / ___/
/ /_/ //  __// / / //  __// /_ / // /__   / ___ | / // /_/ // /_/ // /   / // /_ / / / // / / / / /(__  ) 
\____/ \___//_/ /_/ \___/ \__//_/ \___/  /_/  |_|/_/ \__, / \____//_/   /_/ \__//_/ /_//_/ /_/ /_//____/  
                                                    /____/                                                
"""


# ====================== Adjustable Variables ==============================
current_data_set = "breastCancer"
mutation_rate = .5
cross_over_prob = .5
learning_rate = 1e-3
maxItter = 1
batch_size = .2
population_size = 20
# ===========================================================================


# ================ Data pre-processing =================================================
dataRetriever = DataRetriever("../Datasets/metadata.json")
dataRetriever.retrieveData(current_data_set)
dataset = dataRetriever.getDataSet().dropna()


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
nn = NeuralNetwork(datasetEncoded, 2,
                   [6, 16], dataRetriever.getPredictionType(), dataRetriever.getDataClass())

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

population = [NeuralNetwork(datasetEncoded, 0, [], dataRetriever.getPredictionType(), 
                        dataRetriever.getDataClass()) for i in range(population_size)]


i = 0
while stopping_cond() and i < maxItter:
    for i in range(len(population)):
        random_individual_index = rand.sample([x for x in range(population_size) if x != i], 3)
        # Check fitness
        parent_fitness = population[i].differential_evolution_fitness()
        # Create Trial
        trial_matrix = differential_mutation(population[i].weight_matrix, 
                                            population[random_individual_index[0]].weight_matrix, 
                                            population[random_individual_index[1]].weight_matrix, 
                                            population[random_individual_index[2]].weight_matrix,
                                            mutation_rate)

        # Create offspring
        offspring_matrix = differential_binomial_crossover(population[i].weight_matrix, trial_matrix, cross_over_prob)
        child = NeuralNetwork(datasetEncoded, 0, [], dataRetriever.getPredictionType(), 
                        dataRetriever.getDataClass())
        child.weight_matrix = offspring_matrix
        child_fitness = child.differential_evolution_fitness()

        # exit()
        # if parent_fitness < child_fitness:
        #     population[i] = child

    print(population[i].weight_matrix - trial_matrix)



i += 1





# for i in tqdm(range(maxItter)):
#     # We don't call an inital feedforward because backpropagate starts with a feedforward call
#     # batch_size represents the number of data points per batch
#     nn._back_propagate(learning_rate=learning_rate, batch_size=batch_size, cost_func=cost_func[current_data_set])
#     final = nn.test(testEncoded.drop(dataRetriever.getDataClass(), axis=1))
#     correct = 0
#     for i, row in enumerate(final):
#         if row == actual.iloc[i]:
#             correct += 1

#     perf.append(correct/len(testEncoded))

# # ===============================================================

# # ============= Final Neural Network Output ======
# final = nn.test(testEncoded.drop(dataRetriever.getDataClass(), axis=1))
# output = nn._feed_forward(testEncoded.drop(
#     dataRetriever.getDataClass(), axis=1), testing=True)

# actual = testEncoded[dataRetriever.getDataClass()]


# #  ========== Calculate Accuracy ===========
# final = final.reshape(final.shape[0])
# res = final-actual
# perf = np.asarray(perf)
# plt.plot(np.arange(len(perf)), perf)
# plt.title("Accuracy by Iteration for Breast Cancer")
# plt.xlabel("Iterations")
# plt.ylabel("Accuracy")
# plt.show()
