from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.NeuralNetwork.NeuralNetwork import NeuralNetwork

dataRetriever = DataRetriever("../Datasets/metadata.json")
dataRetriever.retrieveData("breastCancer")

nn = NeuralNetwork(dataRetriever.getDataSet(), 0, [0], dataRetriever.getPredictionType(), dataRetriever.getDataClass())