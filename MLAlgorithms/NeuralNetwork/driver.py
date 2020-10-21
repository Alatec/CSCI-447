from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.NeuralNetwork.NeuralNetwork import NeuralNetwork

dataRetriever = DataRetriever("../Datasets/metadata.json")
dataRetriever.retrieveData("breastCancer")
dataset = dataRetriever.getDataSet().dropna()

dataset = (dataset-dataset.mean())/dataset.std()


nn = NeuralNetwork(dataset, 5, [7,2,4,5,8], dataRetriever.getPredictionType(), dataRetriever.getDataClass())

final_output = nn._feed_forward()
print("Mean: ", final_output.mean())