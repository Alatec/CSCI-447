from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.NeuralNetwork.NeuralNetwork import NeuralNetwork
from MLAlgorithms.Utils.OneHotEncoder import OneHotEncoder
from tqdm import tqdm

dataRetriever = DataRetriever("../Datasets/metadata.json")
dataRetriever.retrieveData("breastCancer")
dataset = dataRetriever.getDataSet().dropna()

dataset = dataset.reset_index(drop=True)
dataset = (dataset-dataset.mean())/dataset.std()
ohe = OneHotEncoder()
datasetEncoded = ohe.oneHotEncoder(dataset, dataRetriever.getDescreteAttributes())




nn = NeuralNetwork(datasetEncoded, 3, [3,2,3], dataRetriever.getPredictionType(), dataRetriever.getDataClass())

# final_output = nn._feed_forward()
# print("Mean: ", final_output.mean())
sample = nn.train_data.sample(frac=0.5)
# initial = nn._feed_forward(sample)
# for i in range(5):
#     nn._backpropagate()

# final = nn._feed_forward()
for i in tqdm(range(1000)):
    nn._backpropagate(batch_size=10)

final = nn._feed_forward(sample)

actual = datasetEncoded.loc[sample.index]["class"]
# print(initial[0])
print(final[0])

correct = 0
for i, row in enumerate(final):
    if row.argmax() == actual.iloc[i].argmax():
        correct += 1

acc = correct/len(sample)
"""

for fold in your_moms_folds:
    nn.train(max_iter, batch_size, radius_of_convergence)
    nn.test()


"""
