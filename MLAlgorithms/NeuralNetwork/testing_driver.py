from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.NeuralNetwork.NeuralNetwork import NeuralNetwork
from MLAlgorithms.Utils.OneHotEncoder import OneHotEncoder
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import os


# Grabs the location of the unit test file and sets cwd
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


# ====================== Adjustable Variables ==============================
learning_rate = 1e-5
maxItter = 4000
batch_size = 100
victimDataset = "breastCancer"


# ================ Data pre-processing =================================================
DS_DICT = {"breastCancer": "bin_cross", 
            "glass": "multi_cross",
            "abalone": "gibgib"}


dataRetriever = DataRetriever("../Datasets/metadata.json")
dataRetriever.retrieveData(victimDataset)
dataset = dataRetriever.getDataSet().dropna()


dataset = dataset.reset_index(drop=True)

# This line is used to normalize the data for Forest Fires
# dataset[dataRetriever.getDataClass()] = np.log(dataset[dataRetriever.getDataClass()]+0.1)

dataset[dataRetriever.getContinuousAttributes()] = (dataset[dataRetriever.getContinuousAttributes()]-dataset[dataRetriever.getContinuousAttributes()].mean())/dataset[dataRetriever.getContinuousAttributes()].std()

test_set = dataset.sample(frac=0.1, random_state=69)
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


# ===========================================================================

# ============== Create Neural Network ===========================
# NOTE: As of right now, the Neural Network only works for classification data sets
nn = NeuralNetwork(datasetEncoded, 2, [3,2], dataRetriever.getPredictionType(), dataRetriever.getDataClass())

# ================================================================


# ======================= Train Neural Network ================

perf = []
actual = testEncoded[dataRetriever.getDataClass()]
for i in range(maxItter):
    # We don't call an inital feedforward because backpropagate starts with a feedforward call
    # batch_size represents the number of data points per batch
    nn._back_propagate(learning_rate=learning_rate, batch_size=batch_size, cost_func = DS_DICT[victimDataset])
    final = nn.test(testEncoded.drop(dataRetriever.getDataClass(), axis=1))
    correct = 0
    for i, row in enumerate(final):
        if row == actual.iloc[i]: correct += 1

    # acc = correct/len(test_set)
    # final = final.reshape(final.shape[0])
    # output = ((output - output.mean())/output.std())
    # actual = (actual - actual.mean())/actual.std()
    # tot_ss = ((actual-actual.mean())**2).sum()
    # res_ss = ((actual-final)**2).sum()
    # R2 = 1-(res_ss/tot_ss)
    # rmse = np.sqrt(((actual-final)**2).mean())
    perf.append(correct/len(testEncoded))
    

# ===============================================================

# ============= Final Neural Network Output ======
final = nn.test(testEncoded.drop(dataRetriever.getDataClass(), axis=1))
output = nn._feed_forward(testEncoded.drop(dataRetriever.getDataClass(), axis=1), testing=True)

actual = testEncoded[dataRetriever.getDataClass()]




#  ================================================


#  ========== Calculate Accuracy ===========


# correct = (actual==np.asarray(final)).sum()

# ===================== Classification =================
correct = 0
for i, row in enumerate(final):
    if row == actual.iloc[i]: correct += 1

acc = correct/len(test_set)

print(acc)
# # ============================================

# # ============ Compare Acc to Most Common Class

values = test_set[dataRetriever.getDataClass()].value_counts()

print(f'Accuracy: {acc}')
print(f'Max Class Prior: {values.max()/values.sum()}')
print(f"Class Distribution:\n{values}")
# fig, axes = plt.subplots(2)
# final = final.reshape(final.shape[0])
# res = final-actual
perf = np.asarray(perf)
plt.plot(np.arange(len(perf)), perf)
# plt.title("Accuracy by Iteration for Breast Cancer")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
# axes[0].set_xlabel("Iterations")
# axes[0].set_ylabel("Acc")
# axes[0].set_ylim([0,1])

# axes[1].plot([0,len(res)],[0,0],linestyle='--',color='black')
# axes[1].scatter(np.arange(len(res)), res)

# axes[1].set_ylabel("Error")
plt.show()



# for i, ax in enumerate(axs):
#     ax.hist(output[:,i])

# plt.show()

## ===================== Regression =================
# fig, axs = plt.subplots(3)
# output = output.reshape(output.shape[0])
# # output = ((output - output.mean())/output.std())
# # actual = (actual - actual.mean())/actual.std()
# rmse =(actual-output)


# # plt.hist(rmse)
# axs[0].hist(actual, label="Actual", alpha=0.5)
# axs[1].hist(output, label="Predicted", alpha=0.5)
# # axs[1].hist(rmse)
# # axs[0].legend()
# axs[2].scatter(actual, output-actual)
# plt.show()


# This section of the driver is used to store parameters that seemed to perform well

#Forest Fires
# learning_rate = 0.17
# maxItter = 500
# batch_size = 40

# nn = NeuralNetwork(datasetEncoded, 3, [3,16,20], dataRetriever.getPredictionType(), dataRetriever.getDataClass())