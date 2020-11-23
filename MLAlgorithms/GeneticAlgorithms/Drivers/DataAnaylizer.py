import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# import ray
# ray.init(num_cpus=11)

file_list = glob.glob('../DataDump/*.json')


def consolidate_data(algorithm, dataset, num_layers):
    fit_glob_string = f"../DataDump/{algorithm}/{dataset}_layer{num_layers}_*fit*"
    out_glob_string = f"../DataDump/{algorithm}/{dataset}_layer{num_layers}_*out*"

    fit_glob_files = glob.glob(fit_glob_string)
    out_glob_files = glob.glob(out_glob_string)

    df = pd.read_csv(fit_glob_files[0])


    fitnesses = np.zeros((len(fit_glob_files), len(df)))

    for i, file_name in enumerate(fit_glob_files):
        df = pd.read_csv(file_name)
        fitnesses[i] = df["Mean_Fitness"]

    means = fitnesses.mean(axis=0)
    stds = fitnesses.std(axis=0)

    iterations = np.arange(fitnesses.shape[1])
    plt.fill_between(iterations, means-stds, means+stds, alpha=0.5)
    plt.plot(iterations, means)
    
    plt.savefig(f"../PlotDump/{dataset}_test.png")


for file_name in file_list:
    base_name = os.path.basename(file_name)
    base_name = os.path.splitext(base_name)[0].split("_")

    algorithm = base_name[0]
    dataset = base_name[1]
    num_layers = int(base_name[2][-1])

    print(f'{algorithm} {dataset} {num_layers}')
    consolidate_data(algorithm, dataset, num_layers)
    break







