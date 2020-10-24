from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.NeuralNetwork.NeuralNetwork import NeuralNetwork
from MLAlgorithms.Utils.OneHotEncoder import OneHotEncoder

dataRetriever = DataRetriever("../Datasets/metadata.json")
dataRetriever.retrieveData("breastCancer")
dataset = dataRetriever.getDataSet().dropna()

# dataset = (dataset-dataset.mean())/dataset.std()
ohe = OneHotEncoder()
datasetEncoded = ohe.oneHotEncoder(dataset, dataRetriever.getDescreteAttributes())
print(datasetEncoded)



nn = NeuralNetwork(datasetEncoded, 5, [7,3,4,5,8], dataRetriever.getPredictionType(), dataRetriever.getDataClass())

# final_output = nn._feed_forward()
# print("Mean: ", final_output.mean())
nn._backpropagate()


"""

for fold in your_moms_folds:
    nn.train(max_iter, batch_size, radius_of_convergence)
    nn.test()


"""
