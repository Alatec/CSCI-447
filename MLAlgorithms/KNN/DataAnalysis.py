import json
import numpy as np
import pandas as pd

import matplotlib
# print(matplotlib.rcsetup.interactive_bk)
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

classification_sets = {}
classification_files = [
    "PerformanceOutput/imageSegPerfWClust.json",
    "PerformanceOutput/glassPerfWClust.json",
    "PerformanceOutput/votePerfWClust.json",
    ]

output_data = {}
names = ["Image Segmentation","Glass","Vote"]
for f, name in zip(classification_files, names):
    f1 = open(f,'r')
    classification_sets[name] = json.load(f1)

rows_list = []
for key, value in classification_sets.items():
    
    
    for fold_num, fold in value.items():
        for k, data in fold.items():
            for alg, metrics in data.items():
                row_dict = {
                    "dataset":key,
                    "k": k,
                    "alg":alg,
                    "f1": metrics["F1"]
                }
                rows_list.append(row_dict)

# dataset_name = "Vote"
for dataset_name in names:
    perf_data = pd.DataFrame(rows_list)
    perf_pivot = pd.pivot_table(perf_data[perf_data["dataset"]==dataset_name], "f1", ["k", "alg"], aggfunc=np.mean)

    graph_data = {}

    for index, row in perf_pivot.iterrows():
        if index[1] not in graph_data:
            graph_data[index[1]] = {}
            graph_data[index[1]]["f1"] = [row["f1"]]
            graph_data[index[1]]["k"] = [int(index[0])]
        else:
            graph_data[index[1]]["f1"].append(row["f1"])
            graph_data[index[1]]["k"].append(int(index[0]))

    for key, value in graph_data.items():
        x = np.asarray(value["k"])
        y = np.asarray(value["f1"])[np.argsort(x)]
        x = np.sort(x)
        plt.plot(x, y, label=key)

    plt.title(dataset_name + " Accurracy by K")
    plt.ylim([0,1])
    plt.legend()
    plt.xlabel("K value")
    plt.ylabel("Accuracy")
    plt.savefig(f"Figures/{dataset_name}_Acc.png")
    plt.clf()
