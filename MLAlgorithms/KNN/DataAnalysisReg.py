import json
import numpy as np
import pandas as pd

import matplotlib
# print(matplotlib.rcsetup.interactive_bk)
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

classification_sets = {}
classification_files = [
    "PerformanceOutput/abalonePerfWClust.json",
    "PerformanceOutput/forestFiresPerfWClust.json",
    "PerformanceOutput/computerHardwarePerfWClust.json",
    ]

output_data = {}

for f, name in zip(classification_files, ["abalone","forestFires","Hardware"]):
    f1 = open(f,'r')
    classification_sets[name] = json.load(f1)

rows_list = []
for key, value in classification_sets.items():

    for fold_num, fold in value.items():
        k_vals = list(fold.keys())
        # print(k_vals)
        for k in k_vals:
            if k == 'mean': continue
            for alg, metrics in fold[k].items():
                row_dict = {
                    "dataset":key,
                    "k": k,
                    "alg":alg,
                    "R2": metrics["R2"]
                }
                rows_list.append(row_dict)

dataset_name = "Hardware"
perf_data = pd.DataFrame(rows_list)
perf_pivot = pd.pivot_table(perf_data[perf_data["dataset"]==dataset_name], "R2", ["k", "alg"], aggfunc=np.mean)

graph_data = {}

for index, row in perf_pivot.iterrows():
    if index[1] not in graph_data:
        graph_data[index[1]] = {}
        graph_data[index[1]]["R2"] = [row["R2"]]
        graph_data[index[1]]["k"] = [int(index[0])]
    else:
        graph_data[index[1]]["R2"].append(row["R2"])
        graph_data[index[1]]["k"].append(int(index[0]))

for key, value in graph_data.items():
    x = np.asarray(value["k"])
    y = np.asarray(value["R2"])[np.argsort(x)]
    x = np.sort(x)
    plt.plot(x, y, label=key)

plt.title(dataset_name)
plt.ylim([0,1])
plt.legend()
plt.show()