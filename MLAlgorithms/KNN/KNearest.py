import numpy as np
import pandas as pd

import matplotlib
print(matplotlib.rcsetup.interactive_bk)
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.KFolds import KFolds
from MLAlgorithms.Utils.ClassifierAnalyzer import ClassifierAnalyzer
from MLAlgorithms.Utils.NumbaFunctions import calculate_euclid_distances

class KNearestNeighbor:



    def __init__(self, test, train, k, unknown_col='class', metric='euclid', classification=True):

        self.test_data = test
        self.train_data = train
        self.k = k
        self.metric = metric
        self.classification = classification

        #This way the unknowns can be passed in as either part of the data frame or as a separate list
        list_types = (list, tuple, np.ndarray, pd.Series)
        if isinstance(unknown_col, list_types):
            self.unknown_col = pd.Series(unknown_col)
        elif isinstance(unknown_col, str):
            self.unknown_col = self.train_data[unknown_col][:]
            self.train_data = self.train_data.drop([unknown_col], axis=1)


        if metric == 'euclid': 
            self.neighbors = calculate_euclid_distances(self.test_data.to_numpy(dtype=np.float64), self.train_data.to_numpy(dtype=np.float64))
        else:
            pass
            # raiasdfse NotIsdfplementedError

    def get_neighbors(self, test_index):      
        """Returns the k nearest neighbors to test index

        Args:
            test_index (int): [description]

        Returns:
            [type]: [description]
        """

        return self.neighbors[test_index].argsort()[:self.k]
    
    #
    def get_most_common_class_apply(self, neighbors_list, augmented=False):
        
        k_neighbors = neighbors_list.argsort()[:self.k]
        unknowns = self.unknown_col.loc[k_neighbors]
        if augmented:
            return (unknowns.iloc[0]==unknowns.iloc[1])
        else:
            return unknowns.value_counts().idxmax()


    def test(self, augmented=False):
        return np.apply_along_axis(self.get_most_common_class_apply,0,self.neighbors,augmented=augmented)

    # def augmented_test(self):
    #     two_neighbors = self.neighbors


class EditedKNN(KNearestNeighbor):
    def __init__(self, test, validation, train, k, unknown_col='class', val_unknown_col='class', metric='euclid', classification=True):
        super().__init__(test, train, k, unknown_col='class', metric='euclid', classification=True)


        self.validation = validation
        list_types = (list, tuple, np.ndarray, pd.Series)
        if isinstance(val_unknown_col, list_types):
            self.val_unknown_col = pd.Series(val_unknown_col)
        elif isinstance(val_unknown_col, str):
            self.val_unknown_col = self.validation[val_unknown_col][:]
            self.validation = self.validation.drop([val_unknown_col], axis=1)
    def train(self, acc_delta=0.01, max_iter=1000, remove_correct=True):

        validation_KNN = KNearestNeighbor(self.validation, self.train_data, self.k, unknown_col=self.unknown_col, 
                                        metric=self.metric, classification=self.classification)
        intial_validation_performance = (validation_KNN.test() == self.val_unknown_col).sum()/len(self.validation)
        curr_performance = intial_validation_performance
        prev_performance = 0.01
        curr_iter = 0
        # 3 Exit conditions: Performance is worse than when we started, performance hasn't increased, hit iteration limit
        while curr_performance >= (1+acc_delta)*prev_performance and curr_iter < max_iter:
            prev_performance = curr_performance
            testing_KNN = KNearestNeighbor(self.train_data, self.train_data, 2, unknown_col=self.unknown_col, 
                                        metric=self.metric, classification=self.classification)
            correct_points = testing_KNN.test(augmented=True)
            points_to_remove = np.argwhere(correct_points==remove_correct).flatten()
            temp_train = self.train_data.drop(points_to_remove, axis=0)
            validation_KNN = KNearestNeighbor(self.validation, temp_train, self.k, unknown_col=self.val_unknown_col, 
                                        metric=self.metric, classification=self.classification)
            curr_performance = (validation_KNN.test() == self.unknown_col).sum()/len(self.validation)
            if curr_performance < intial_validation_performance:
                break
            else:
                self.train_data = temp_train.deep_copy()
            
            curr_iter += 1

        # baseline_acc = classifierAnalyzer.calc_accuracy(method=method)

if __name__ == "__main__":
    dataRetriever = DataRetriever("../Datasets/metadata.json")
    dataRetriever.retrieveData("breastCancer")
    data = dataRetriever.getDataSet()
    data = data.dropna()
    data = data.reset_index(drop=True)

    test = data.sample(frac=0.2)
    train = data.drop(test.index)

    validate = test.sample(frac=0.5)
    test = test.drop(validate.index)

    train = train.reset_index(drop=True)
    test  = test.reset_index(drop=True)
    validate = validate.reset_index(drop=True)

    KNN = KNearestNeighbor(test.drop('class', axis=1),train,5)
    EKNN = EditedKNN(test.drop('class', axis=1), validate, train, 5)
    EKNN.train()
    # print(KNN.get_neighbors([5,10]))
    # print(KNN.test(augmented=True))
        