from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.NeuralNetwork.NeuralNetwork import NeuralNetwork
from MLAlgorithms.Utils.OneHotEncoder import OneHotEncoder
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# ================ Data pre-processing =================================================

dataRetriever = DataRetriever("../Datasets/metadata.json")
dataRetriever.retrieveData("glass")
dataset = dataRetriever.getDataSet().dropna()

dataset = dataset.reset_index(drop=True)
dataset[dataRetriever.getContinuousAttributes()] = (dataset[dataRetriever.getContinuousAttributes()]-dataset[dataRetriever.getContinuousAttributes()].mean())/dataset[dataRetriever.getContinuousAttributes()].std()

test_set = dataset.sample(frac=0.1)
train_set = dataset.drop(test_set.index)
test_set = test_set.reset_index(drop=True)
train_set = train_set.reset_index(drop=True)

ohe = OneHotEncoder()
discrete_attr = dataRetriever.getDescreteAttributes()
if dataRetriever.getDataClass() in discrete_attr:
    discrete_attr.remove(dataRetriever.getDataClass())

datasetEncoded = ohe.train_fit(train_set, dataRetriever.getDescreteAttributes())
testEncoded = ohe.fit(test_set)


# =======================================================================================

# ====================== Adjustable Variables ==============================
learning_rate = 0.5
maxItter = 200
batch_size = 50
# ===========================================================================

# ============== Create Neural Network ===========================
# NOTE: As of right now, the Neural Network only works for classification data sets
nn = NeuralNetwork(datasetEncoded, 2, [5,7], dataRetriever.getPredictionType(), dataRetriever.getDataClass())

# ================================================================


# ======================= Train Neural Network ================


cost_func = []
for i in range(maxItter):
    # We don't call an inital feedforward because backpropagate starts with a feedforward call
    # batch_size represents the number of data points per batch
    output = nn._back_propagate(learning_rate=learning_rate, batch_size=batch_size)
    # print(output.sum().sum())
    # max_weight = nn.weight_matrix.max()
    # if np.isnan(max_weight):
    #     break
    # else:
    #     print(f'{i} : {max_weight}')

# ===============================================================

# ============= Final Neural Network Output ======
final = nn.test(testEncoded.drop(dataRetriever.getDataClass(), axis=1))
output = nn._feed_forward(testEncoded.drop(dataRetriever.getDataClass(), axis=1), testing=True)

actual = testEncoded[dataRetriever.getDataClass()]


#  ================================================


#  ========== Calculate Accuracy ===========


# correct = (actual==np.asarray(final)).sum()

correct = 0
for i, row in enumerate(final):
    if row == actual.iloc[i]: correct += 1

acc = correct/len(test_set)
# ============================================

# ============ Compare Acc to Most Common Class

values = train_set[dataRetriever.getDataClass()].value_counts()

print(f'Accuracy: {acc}')
print(f'Max Class Prior: {values.max()/values.sum()}')
print(f"Class Distribution:\n{values}")
# plt.hist(output)
# plt.show()
