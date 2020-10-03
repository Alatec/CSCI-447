from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.KNN.KMeansClustering import KMeans

# dataSets = ["vote","imageSegmentation", "computerHardware", ]
#             # "forestFires", "abalone"]

data = DataRetriever("../Datasets/metadata.json")



# for item in dataSets:
print(f"Creating CSV for vote")
data.retrieveData("vote")

maxItter = 100
kValue = 3
centroids = KMeans(data.getDataSet(), data.getDataClass(), data.getDescreteAttributes(),
                   data.getContinuousAttributes(), data.getPredictionType(), kValue, maxItter)

centroids.to_csv('./CSVOutput/' + "vote" + 'KMeansClustered.csv', index=False)
print(f"CSV for vote has been created!")
