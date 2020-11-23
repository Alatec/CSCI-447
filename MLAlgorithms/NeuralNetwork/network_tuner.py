from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.NeuralNetwork.NeuralNetwork import NeuralNetwork
from MLAlgorithms.Utils.OneHotEncoder import OneHotEncoder
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import itertools


def network_tuner(*nodes_per_hidden_layer):
    """
    This function is used to calcuate the optimal network architecture
    The user should input the dataset they would like to operate with and change the performance metric in accordance to the data set type IE regression or classification 

    """
    
    MSEs = []

    bestNetwork = {}
    learning_rate = 0.0001
    maxItter = 500
    batch_size = .5

    dataRetriever = DataRetriever("../Datasets/metadata.json")
    dataRetriever.retrieveData("glass")
    dataset = dataRetriever.getDataSet().dropna()


    dataset = dataset.reset_index(drop=True)

    # This line is used to normalize the data for Forest Fires
    # dataset[dataRetriever.getDataClass()] = np.log(dataset[dataRetriever.getDataClass()]+0.1)

    dataset[dataRetriever.getContinuousAttributes()] = (dataset[dataRetriever.getContinuousAttributes()]-dataset[dataRetriever.getContinuousAttributes()].mean())/dataset[dataRetriever.getContinuousAttributes()].std()

    test_set = dataset.sample(frac=0.1, random_state=69)
    train_set = dataset.drop(test_set.index)
    test_set = test_set.reset_index(drop=True)
    train_set = train_set.reset_index(drop=True)

    ohe = OneHotEncoder()
    discrete_attr = dataRetriever.getDescreteAttributes()
    if dataRetriever.getDataClass() in discrete_attr:
        discrete_attr.remove(dataRetriever.getDataClass())

    datasetEncoded = ohe.train_fit(train_set, dataRetriever.getDescreteAttributes())
    testEncoded = ohe.fit(test_set)


    output = None
    nn = NeuralNetwork(datasetEncoded, 0, [], dataRetriever.getPredictionType(), dataRetriever.getDataClass())
    for i in range(maxItter):
        # We don't call an inital feedforward because backpropagate starts with a feedforward call
        # batch_size represents the number of data points per batch
        output = nn._back_propagate(learning_rate=learning_rate, batch_size=batch_size)


    final = nn.test(testEncoded.drop(dataRetriever.getDataClass(), axis=1))
    output = nn._feed_forward(testEncoded.drop(dataRetriever.getDataClass(), axis=1), testing=True)
    actual = testEncoded[dataRetriever.getDataClass()]


    ## ===================== Classification =================
    correct = 0
    acc = 0
    for i, row in enumerate(final):
        if row == actual.iloc[i]: correct += 1


    # final = final.reshape(final.shape[0])

    # MSE = ((actual-final)**2).mean()
    # MSEs.append(MSE)
    bestNetwork['network'] = nn
    bestNetwork['acc'] = acc
    bestNetwork['arc'] = [0]
    # # ============================================

    # # ============ Compare Acc to Most Common Class

    values = test_set[dataRetriever.getDataClass()].value_counts()


    # USED FOR CLASSIFICATION
    # print(f'Accuracy: {acc}')
    # print(f'Max Class Prior: {values.max()/values.sum()}')
    # print(f"Class Distribution:\n{values}")
    # print("Final: ", final)
    # print("Actual: ", list(actual))
    # print()



    numOfLayer = len(nodes_per_hidden_layer)
    print("Number of Hidden Layers: ", numOfLayer)
    for layer in range(numOfLayer):
        print(f"Layer Number: {layer + 1}")
        combinations = list(itertools.product(*nodes_per_hidden_layer[:layer+1]))

        for combo in combinations:

            output = None
            print("Node Combination: ",list(combo))
            print(combo)

            nn = NeuralNetwork(datasetEncoded, layer, list(combo), dataRetriever.getPredictionType(), dataRetriever.getDataClass())
            for i in range(maxItter):
                # We don't call an inital feedforward because backpropagate starts with a feedforward call
                # batch_size represents the number of data points per batch
                output = nn._back_propagate(learning_rate=learning_rate, batch_size=batch_size)

            final = nn.test(testEncoded.drop(dataRetriever.getDataClass(), axis=1))
            output = nn._feed_forward(testEncoded.drop(dataRetriever.getDataClass(), axis=1), testing=True)
            actual = testEncoded[dataRetriever.getDataClass()]

            ## ===================== Classification =================
            correct = 0
            acc = 0
            for i, row in enumerate(final):
                if row == actual.iloc[i]: correct += 1

            acc = correct/len(test_set)
            # # # ============================================

            # # # ============ Compare Acc to Most Common Class

            values = test_set[dataRetriever.getDataClass()].value_counts()

            # USED FOR CLASSIFICATION
            # print(f'Accuracy: {acc}')
            # print(f'Max Class Prior: {values.max()/values.sum()}')
            # # print(f"Class Distribution:\n{values}")
            # print("Final: ", final)
            # print("Actual: ", list(actual))
            # print()

            if acc > bestNetwork['acc']:
                bestNetwork['network'] = nn
                bestNetwork['acc'] = acc
                bestNetwork['arc'] = combo

            # final = final.reshape(final.shape[0])

            # MSE = ((actual-final)**2).mean()
            # MSEs.append(MSE)
            # if MSE < bestNetwork['acc']:
            #     bestNetwork['network'] = nn
            #     bestNetwork['acc'] = MSE
            #     bestNetwork['arc'] = combo

            



    return bestNetwork#, MSEs


arr1 = [*range(1, 11, 1)]
arr2 = [*range(1, 11, 1)]

bestArchitecture = network_tuner(arr1, arr2)

print(bestArchitecture)





