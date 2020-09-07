from MLAlgorithms.NaiveBayes.naiveBayes import NaiveBayes
from MLAlgorithms.Utils.DataRetriever import DataRetriever

dataRetriever = DataRetriever("../Datasets/metadata.json")

print(dataRetriever.retrieveData('breastCancer'))

# For train. test in KFOLD:
#   binData()
#   NormalizeData
#   Naivebayes

naiveBayes = NaiveBayes()