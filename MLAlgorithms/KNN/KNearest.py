import numpy as np
import pandas as pd

# import matplotlib
# print(matplotlib.rcsetup.interactive_bk)
# matplotlib.use('WebAgg')
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
    def get_most_common_class_apply(self, neighbors_list, augmented=False, verbose=False):
        
        k_neighbors = neighbors_list.argsort()[:self.k]
        if verbose:
            print("K Neighbors", k_neighbors)
            print(self.unknown_col.shape)
        unknowns = self.unknown_col.loc[k_neighbors]
        if augmented:
            return (unknowns.iloc[0]==unknowns.iloc[1])
        else:
            return unknowns.value_counts().idxmax()


    def test(self, augmented=False, verbose=False):
        return np.apply_along_axis(self.get_most_common_class_apply,1,self.neighbors,augmented=augmented,verbose=verbose)

    # def augmented_test(self):
    #     two_neighbors = self.neighbors


class EditedKNN(KNearestNeighbor):
    def __init__(self, test, validation, train, k, unknown_col='class', val_unknown_col='class', metric='euclid', classification=True):
        super().__init__(test, train, k, unknown_col=unknown_col, metric=metric, classification=classification)


        self.validation = validation
        # print("Validation shape:", self.validation.shape)
        list_types = (list, tuple, np.ndarray, pd.Series)
        if isinstance(val_unknown_col, list_types):
            self.val_unknown_col = pd.Series(val_unknown_col)
        elif isinstance(val_unknown_col, str):
            self.val_unknown_col = self.validation[val_unknown_col][:]
            self.validation = self.validation.drop([val_unknown_col], axis=1)

        # print("Val unknown size", self.val_unknown_col.shape)

    def train(self, acc_delta=0.01, max_iter=1000, remove_correct=False):
        # print(self.validation)
        validation_KNN = KNearestNeighbor(self.validation, self.train_data, 1, unknown_col=self.unknown_col, 
                                        metric=self.metric, classification=self.classification)
        print("Valid shape", self.validation.shape)
        print("Train shape", self.train_data.shape)
        print("Val unknown shape", self.val_unknown_col.shape)
        print("Val test shape", len(validation_KNN.test()))
        intial_validation_performance = (validation_KNN.test() == self.val_unknown_col).sum()
        intial_validation_performance /= len(self.validation)
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
            temp_train = self.train_data.drop(points_to_remove, axis=0).reset_index(drop=True)
            temp_unknown_col = self.unknown_col.drop(points_to_remove).reset_index(drop=True)
            del(validation_KNN)
            validation_KNN = KNearestNeighbor(self.validation, temp_train, 1, unknown_col=temp_unknown_col, 
                                        metric=self.metric, classification=self.classification)
            curr_performance = (validation_KNN.test() == self.val_unknown_col).sum()/len(self.validation)
            print(f"Iter:{curr_iter}, Curr Perf: {curr_performance},  Prev Perf: {prev_performance}")
            print(len(points_to_remove))
            if curr_performance < intial_validation_performance and curr_iter > 0:
                break
            else:
                self.train_data = temp_train.copy(deep=True)
                self.unknown_col = temp_unknown_col.copy(deep=True)
            
            
            curr_iter += 1
            

        # baseline_acc = classifierAnalyzer.calc_accuracy(method=method)


class CondensedKNN(KNearestNeighbor):
    def __init__(self, test, train, k, unknown_col='class', metric='euclid', classification=True):
        super().__init__(test, train, k, unknown_col=unknown_col, metric=metric, classification=classification)

        # print("Val unknown size", self.val_unknown_col.shape)

    def train(self, acc_delta=0.01, max_iter=1000, remove_correct=False):
        # print(self.validation)
        np.random.seed(0)

        curr_iter = 0

        #Initialize X and Z sets
        x_set = np.ones_like(self.unknown_col)
        z_set = np.zeros_like(x_set)

        #Initialize Z with a random subset of X
        z_set[list(np.random.choice(self.train_data.index, self.unknown_col.nunique(), replace=False))] = 1
        x_set[z_set==1] = 0
        print("Z")
        print(z_set, z_set.sum())
        print("X")
        print(x_set, x_set.sum())
        prev_z_len = 0

        # 3 Exit conditions: Performance is worse than when we started, performance hasn't increased, hit iteration limit
        while prev_z_len * 1.1 < z_set.sum() and curr_iter < max_iter:
            prev_z_len = z_set.sum()

            testing_KNN = KNearestNeighbor(self.train_data.loc[x_set==1], self.train_data[z_set==1], 2, unknown_col=self.unknown_col, 
                                        metric=self.metric, classification=self.classification)

            #Numpy array containing points in the X set classified correctly
            correct_points = (testing_KNN.test()==self.unknown_col[x_set==1])
            # print(correct_points)
            tested_indexes = self.train_data.index[x_set==1]
            print("Z size:", z_set.sum())
            for index in tested_indexes[correct_points==False]:
                z_set[index] = 1
                x_set[index] = 0

            #Break if either set is empty
            if x_set.sum() * z_set.sum() == 0: break

           
            
            
            curr_iter += 1

        self.train_data = self.train_data.drop(self.train_data.index[x_set==1])
        print(curr_iter)
            

if __name__ == "__main__":
    dataRetriever = DataRetriever("../Datasets/metadata.json")
    dataRetriever.retrieveData("abalone")
    data = dataRetriever.getDataSet()
    data = data.dropna()
    data = data.reset_index(drop=True)
    data = data.drop('sex', axis=1)

    test = data.sample(frac=0.2, random_state=17)
    train = data.drop(test.index)

    validate = test.sample(frac=0.5, random_state=13)
    test = test.drop(validate.index)

    train = train.reset_index(drop=True)
    test  = test.reset_index(drop=True)
    validate = validate.reset_index(drop=True)

    class_col = 'rings'

    # KNN = KNearestNeighbor(test.drop(class_col, axis=1),train,5, unknown_col=class_col)
    # print("KNN test length", len(KNN.test()))
    # print("Test set length", test.shape)
    CKNN = CondensedKNN(test.drop(class_col, axis=1), train, 5, unknown_col=class_col)
    CKNN.train()
    # print(KNN.get_neighbors([5,10]))
    # print(KNN.test(augmented=True))