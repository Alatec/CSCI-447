from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.KNN.KMeansClustering import KMeans

dataSets = ["vote","imageSegmentation", "computerHardware", ]
            # "forestFires", "abalone"]

data = DataRetriever("../Datasets/metadata.json")



for item in dataSets:
    print(f"Creating CSV for {item}")
    data.retrieveData(item)

    maxItter = 5
    kValue = 3
    centroids = KMeans(data.getDataSet(), data.getDataClass(), data.getDescreteAttributes(),
                       data.getContinuousAttributes(), data.getPredictionType(), kValue, maxItter)

    centroids.to_csv('./CSVOutput/' + item + 'KMeansClustered.csv', index=False)
    print(f"CSV for {item} has been created!")
