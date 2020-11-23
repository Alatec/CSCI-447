import json

from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np



unshuffledData = None
shuffledData = None

with open("Nonshuffled.json", "r") as f:
    unshuffledData = json.load(f)

with open("Shuffled.json", "r") as f:
    shuffledData = json.load(f)

unShuffledList = []
shuffledList = []

recallUnShuffledList = []
recallShuffledList = []

precisionUnShuffledList = []
precisionShuffledList = []

f1MeanUnShuffledDict = {}
f1MeanShuffledDict = {}

"""
for each dataset in unshuffled:
    grab the k1score mean

for each dataset in shuffled:
    grab the k1score mean
"""

for dataSet, value in unshuffledData.items():
    f1MeanUnShuffledDict[dataSet] = []
    for fold, metrics in value.items():
        f1MeanUnShuffledDict[dataSet].append(metrics["F1Score"])
        unShuffledList.append(metrics["F1Score"])
        recallUnShuffledList.append(metrics["Recall"])
        precisionUnShuffledList.append(metrics["Precision"])

    f1MeanUnShuffledDict[dataSet] = np.array(f1MeanUnShuffledDict[dataSet]).mean()

for dataSet, value in shuffledData.items():
    f1MeanShuffledDict[dataSet] = []
    for fold, metrics in value.items():
        f1MeanShuffledDict[dataSet].append(metrics["F1Score"])
        shuffledList.append(metrics["F1Score"])
        recallShuffledList.append(metrics["Recall"])
        precisionShuffledList.append(metrics["Precision"])

    f1MeanShuffledDict[dataSet] = np.array(f1MeanShuffledDict[dataSet]).mean()


# print(f1MeanUnShuffledDict)
# print(f1MeanShuffledDict)

# N = len(f1MeanShuffledDict.keys())
# ind = np.arange(N)
# width = .35
# fig, ax = plt.subplots()

# p1 = ax.bar(ind, f1MeanUnShuffledDict.values(), width, 0)


# p2 = ax.bar(ind + width, f1MeanShuffledDict.values(), width, 0)



# ax.legend((p1[0], p2[0]), ("UnShuffled", "Shuffled"))
# ax.set_title("Average F1 Scores by Dataset and Shuffle Type")
# ax.set_xticks(ind + width / 2)
# ax.set_xticklabels(f1MeanShuffledDict.keys())
# ax.autoscale_view()

# plt.show()

#######################################################################
# recallUnShuffledList = np.array(recallUnShuffledList)
# precisionUnShuffledList = np.array(precisionUnShuffledList)

# recallShuffledList = np.array(recallShuffledList)
# precisionShuffledList = np.array(precisionShuffledList)

# meanRecallUnShuffledList = recallUnShuffledList.mean()
# meanPrecisionUnShuffledList = precisionUnShuffledList.mean()

# meanRecallShuffledList = recallShuffledList.mean()
# meanPrecisionShuffledList = precisionShuffledList.mean()

# print(f"Mean of the Recall for the unshuffled list: {meanRecallUnShuffledList}\n"
#       f"Mean of the Precision for unshuffled list: {meanPrecisionUnShuffledList}\n"
#       f"Mean of the Recall for the shuffled list: {meanRecallShuffledList}\n"
#       f"Mean of the Precision for the shuffled list: {meanPrecisionShuffledList}")

###############################################################################
unShuffledList = np.array(unShuffledList)
shuffledList = np.array(shuffledList)

mu1 = unShuffledList.mean()
mu2 = shuffledList.mean()

sd1 = unShuffledList.std()
sd2 = shuffledList.std()

print(f"M1 {mu1} \nM2 {mu2}\nSD1 {sd1}\nSD2 {sd2}\nMU1 - MU2 {mu1 - mu2}")

scale = np.sqrt((sd1 + sd2) / 50)
zStat = (mu1 - mu2) / scale

print(sd1)
print(sd2)

print(1-norm.pdf(zStat))

# plt.hist(unShuffledList, bins=15, alpha=0.5, label="UnShuffled Features")
# plt.hist(shuffledList, bins=15, alpha=0.5, label="Shuffled Features")
# plt.title("Average F1 Scores by Fold")
# plt.legend()
# plt.show()

###################################################################################

