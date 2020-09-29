import pandas as pd
import numpy as np
import matplotlib
print(matplotlib.rcsetup.interactive_bk)
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.KFolds import KFolds
import numba

dataRetriever = DataRetriever("../Datasets/metadata.json")
dataRetriever.retrieveData("breastCancer")
data = dataRetriever.getDataSet()
data = data.dropna()
data = data.reset_index(drop=True)


times = []
rows_list = []
distances = np.zeros((len(data), len(data)), dtype=np.float)
for index, row in tqdm(data.iterrows(), total=len(data)):
    for index2, row2 in data.iterrows():
        distances[index, index2] = ((row2-row)**2).sum()

    
@numba.njit
def calc_distances(numpy_array):
    distances = np.zeros((numpy_array.shape[0], numpy_array.shape[0]), dtype=np.float64)

    for i in range(numpy_array.shape[0]):
        for j in range(numpy_array.shape[0]):
            distances[i,j] = ((numpy_array[i]-numpy_array[j])**2).sum()
           
        
    return distances

start = time.time()
for i in tqdm(range(10)):
    distances2 = calc_distances(data.to_numpy())

total_time = time.time() - start

print(total_time/10)
# # distances = pd.DataFrame(rows_list)
# print(distances)
# # for test, train in KFolds(data, 10):


# # plt.scatter(data["uniformityOfCellSize"], data["uniformityOfCellShape"])
# # plt.show()
# sd_norm = (data-data.mean())/data.std()
# sd_norm = sd_norm.drop('class', axis=1)
# sd_cov = sd_norm.cov()

