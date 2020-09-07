from MLAlgorithms.NaiveBayes.naiveBayes import NaiveBayes
from MLAlgorithms.Utils.DataRetriever import DataRetriever



# For train. test in KFOLD:
#   binData()
#   NormalizeData
#   Naivebayes

dataRetriever = DataRetriever("../Datasets/metadata.json")
dataRetriever.retrieveData("breastCancer")
naiveBayes = NaiveBayes(dataRetriever.getDataSet(), dataRetriever.getDataClass())
