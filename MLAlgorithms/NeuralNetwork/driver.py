from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.NeuralNetwork.NeuralNetwork import NeuralNetwork
from MLAlgorithms.Utils.OneHotEncoder import OneHotEncoder
from tqdm import tqdm
import numpy as np

# ================ Data pre-processing =================================================

dataRetriever = DataRetriever("../Datasets/metadata.json")
dataRetriever.retrieveData("breastCancer")
dataset = dataRetriever.getDataSet().dropna()

dataset = dataset.reset_index(drop=True)
dataset = (dataset-dataset.mean())/dataset.std()
ohe = OneHotEncoder()
datasetEncoded = ohe.oneHotEncoder(dataset, dataRetriever.getDescreteAttributes())

unique = ohe.encodedDict.keys()

# =======================================================================================

# ====================== Adjustable Variables ==============================
learning_rate = 0.01
maxItter = 40
batch_size = 10
# ===========================================================================

# ============== Create Neural Network ===========================
# NOTE: As of right now, the Neural Network only works for classification data sets
nn = NeuralNetwork(dataset, 2, [5, 2], dataRetriever.getPredictionType(), dataRetriever.getDataClass(), is_binary_class=True)

# ================================================================


# ======================= Train Neural Network ================

# Train the Neural Network 1000 times
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
final = nn._feed_forward(nn.train_data)

actual = nn.unknown_col[nn.train_data.index]
# print(initial[0])

#  ================================================


#  ========== Calculate Accuracy ===========

correct = 0
thresh = np.mean(final)
for i, row in enumerate(final):
    if row[0] < thresh and 0 == actual[i]:
        correct += 1
    elif row[0] >= thresh and 1 == actual[i]:
        correct += 1
acc = correct/len(nn.train_data)
# ============================================

# ============ Compare Acc to Most Common Class

values = datasetEncoded[dataRetriever.getDataClass()].value_counts()

print(f'Accuracy: {acc}')
print(f'Max Class Prior: {values.max()/values.sum()}')
print(f"Class Distribution:\n{values}")