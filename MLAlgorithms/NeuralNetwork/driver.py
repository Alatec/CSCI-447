from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.NeuralNetwork.NeuralNetwork import NeuralNetwork
from MLAlgorithms.Utils.OneHotEncoder import OneHotEncoder
from tqdm import tqdm

# ================ Data pre-processing =================================================

dataRetriever = DataRetriever("../Datasets/metadata.json")
dataRetriever.retrieveData("breastCancer")
dataset = dataRetriever.getDataSet().dropna()

dataset = dataset.reset_index(drop=True)
dataset = (dataset-dataset.mean())/dataset.std()
ohe = OneHotEncoder()
datasetEncoded = ohe.oneHotEncoder(dataset, dataRetriever.getDescreteAttributes())

# =======================================================================================

# ====================== Adjsutable Variables ==============================
learning_rate = .01
maxItter = 3
# ===========================================================================

# ============== Create Neural Network ===========================
# NOTE: As of right now, the Neural Network only works for classification data sets
nn = NeuralNetwork(datasetEncoded, 3, [3,2,3], dataRetriever.getPredictionType(), dataRetriever.getDataClass())

# ================================================================


# ======================= Train Neural Network ================

# Train the Neural Network 1000 times
for i in tqdm(range(maxItter)):
    # We don't call an inital feedforward because backpropagate starts with a feedforward call
    # batch_size represents the number of data points per batch
    nn._back_propagate(learning_rate=learning_rate, batch_size=100)

# ===============================================================

# ============= Final Neural Network Output ======
final = nn._feed_forward(nn.train_data)

actual = datasetEncoded.loc[nn.train_data.index][dataRetriever.getDataClass()]
# print(initial[0])
print(final[0])
#  ================================================


#  ========== Calculate Accuracy ===========

correct = 0
for i, row in enumerate(final):
    if row.argmax() == actual.iloc[i].argmax():
        correct += 1
acc = correct/len(nn.train_data)
# ============================================

# ============ Compare Acc to Most Common Class

values = datasetEncoded[dataRetriever.getDataClass()].value_counts()

print(f'Accuracy: {acc}')
print(f'Max Class Prior: {values.max()/values.sum()}')
print(f"Class Distribution:\n{values}")