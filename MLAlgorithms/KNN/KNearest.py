import numpy as np
import pandas as pd

import matplotlib
# print(matplotlib.rcsetup.interactive_bk)
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.KFolds import KFolds
from MLAlgorithms.Utils.ClassifierAnalyzer import ClassifierAnalyzer
from MLAlgorithms.Utils.NumbaFunctions import calculate_euclid_distances
from MLAlgorithms.Utils.DistanceMatrix import DistanceMatrix
from MLAlgorithms.Utils.StandardNormalizer import StandardNormalizer
from MLAlgorithms.Utils.RangeNormalizer import RangeNormalizer

class KNearestNeighbor:



    def __init__(self, test, train, k, contAttr, discAttr, unknown_col='class', predictionType="classification"):

        """Creates an instance of a trained NaiveBayes Algorithm

        Args:
            test (DataFrame): The test set
            train (DataFrame): The train set
            k     (int or list of int): value(s) of k to use for KNN
            contAttr (list of string): List of columns containing continuous data
            discAttr     (int or list of int): List of columns containing discrete data
            unknown_col  (string): attribute to be predicted
            prediction_type    (string): classification or regression
        """
        self.test_data = test
        self.train_data = train
        self.k = k
        self.predictionType = predictionType
        self.contAttr = contAttr
        self.discAttr = discAttr

        
        #This way the unknowns can be passed in as either part of the data frame or as a separate list
        #If the unknowns are part of the training set, they are split into their own Series
        list_types = (list, tuple, np.ndarray, pd.Series)
        if isinstance(unknown_col, list_types):
            self.unknown_col = pd.Series(unknown_col)
        elif isinstance(unknown_col, str):
            if unknown_col in self.discAttr:
                self.discAttr.remove(unknown_col)

            if unknown_col in self.contAttr:
                self.contAttr.remove(unknown_col)
            self.unknown_col = self.train_data[unknown_col][:]
            self.train_data = self.train_data.drop(unknown_col, axis=1)

        #Training the VDM requires the unknown column to be part of the set, so it is added to a temporary data frame
        self.unknown_col.reset_index(drop=True)
        self.train_data.reset_index(drop=True)
        temp_train_with_unknown = self.train_data.copy(deep=True)
        temp_train_with_unknown["unknown_col"] = self.unknown_col.values

        #Distance matrix is a tool for creating a weighted sum discrete and continous distances
        self.distance_matrix = DistanceMatrix(self.test_data, temp_train_with_unknown, contAttr, discAttr, len(contAttr), len(discAttr), predictionType, "unknown_col")
        
        #self.neigbors is a len(test)xlen(train) array containing the distance from each point in test to each point in train
        self.neighbors = self.distance_matrix.distanceMatrix 

    def get_neighbors(self, test_index):      
        """Returns the k nearest neighbors to test index

        Args:
            test_index (int): the point in the test set

        Returns:
            list: List containing the indexes of the k closest points in the training set to the test index
        """

        return self.neighbors[test_index].argsort()[:self.k]
    
    #
    def get_most_common_class_apply(self, neighbors_list, augmented=False, verbose=False):
        
        """ Function intended to be applied row-wise to the neigbors list

        Returns
        -------
        neighbors_list
            a 2d matrix containing where each cell describe the distance from the ith test point to the jth train point
        agumented
            whether or not to use a version optimized for augmented KNN (currently only used for Edited KNN)
        verbose
            whether to include additional debug information
        """

        if type(self.k) == list:
            #Run on a list of k to avoid regenerating neighbors
            #The else clause is commented more thoroughly
            return_vals = []
            for k in self.k:
                k_neighbors = neighbors_list.argsort()[:k]
                unknowns = self.unknown_col.iloc[k_neighbors]

                if self.predictionType == "regression":
                    mu = unknowns.mean()
                    var = unknowns.var()

                    #The value of 5 was chosen empirically
                    eps = 5/(var+0.001)

                    kernals = np.exp(-(eps*(np.abs(unknowns-mu)))**2)
                    return_vals.append(((unknowns*kernals)/kernals.sum()).sum())
                else:
                    return_vals.append(unknowns.value_counts().idxmax())
            return np.asarray(return_vals)
        else:

            #Calculate the unknown values associated with the k closest neighbors
            k_neighbors = neighbors_list.argsort()[:self.k]
            unknowns = self.unknown_col.iloc[k_neighbors]

            if self.predictionType == "regression":

                #Use a gaussian kernel to predict regression values
                mu = unknowns.mean()
                var = unknowns.var()

                #Epsilon is based on the variance of the unknowns instead of a fixed value
                #The value of 5 was chosen empirically
                eps = 5/(var+0.001)

                #Calculate the predicted value
                kernals = np.exp(-(eps*(np.abs(unknowns-mu)))**2)
                return ((unknowns*kernals)/kernals.sum()).sum()


            if verbose:
                print("K Neighbors", k_neighbors)
                print(self.unknown_col.shape)
            
            if augmented:
                #Returns a boolean list indicated if the predicted and true class are the same
                return (unknowns.iloc[0]==unknowns.iloc[1])
            else:
                #Return the most common value from the unknowns
                return unknowns.value_counts().idxmax()


    def test(self, augmented=False, verbose=False):
        return np.apply_along_axis(self.get_most_common_class_apply,1,self.neighbors,augmented=augmented,verbose=verbose)



class EditedKNN(KNearestNeighbor):
    def __init__(self, test, validation, train, k, contAttr, discAttr, unknown_col='class', val_unknown_col='class', predictionType="classification"):
        super().__init__(test, train, k, contAttr, discAttr, unknown_col, predictionType)

        #The arguments are the same as KNN but with information for handling the validation set added

        self.validation = validation

        #Similar to the unknown_col in KNN just for val_unknown_col
        list_types = (list, tuple, np.ndarray, pd.Series)
        if isinstance(val_unknown_col, list_types):
            self.val_unknown_col = pd.Series(val_unknown_col)
        elif isinstance(val_unknown_col, str):
            if val_unknown_col in self.discAttr:
                self.discAttr.remove(val_unknown_col)

            if val_unknown_col in self.contAttr:
                self.contAttr.remove(val_unknown_col)

            self.val_unknown_col = self.validation[val_unknown_col][:]
            self.validation = self.validation.drop([val_unknown_col], axis=1)



    def train(self, acc_delta=0.01, max_iter=1000, remove_correct=False):

        validation_KNN = KNearestNeighbor(self.validation, self.train_data, 1,self.contAttr, self.discAttr, unknown_col=self.unknown_col, predictionType=self.predictionType)
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

            #Create an instance of KNN using train_data as both test and train sets
            testing_KNN = KNearestNeighbor(self.train_data, self.train_data, 2, self.contAttr, self.discAttr, unknown_col=self.unknown_col, predictionType=self.predictionType)

            #Boolean list containing whether the point was correctly classified
            correct_points = testing_KNN.test(augmented=True)

            #Indexes predicted incorrectly
            points_to_remove = np.argwhere(correct_points==remove_correct).flatten()

            #Create a temporary training set and compare performance against validation set
            temp_train = self.train_data.drop(points_to_remove, axis=0).reset_index(drop=True)
            temp_unknown_col = self.unknown_col.drop(points_to_remove).reset_index(drop=True)
            del(validation_KNN)
            if len(temp_train) == 0:
                #Break if there's nothing in the temp set
                break
            validation_KNN = KNearestNeighbor(self.validation, temp_train, 1, self.contAttr, self.discAttr, unknown_col=temp_unknown_col, predictionType=self.predictionType)
            curr_performance = (validation_KNN.test() == self.val_unknown_col).sum()/len(self.validation)
            print(f"Iter: {curr_iter}, Curr Perf: {curr_performance},  Prev Perf: {prev_performance}")
            print(len(points_to_remove))
            if curr_performance < intial_validation_performance and curr_iter > 0:
                break
            else:
                #Assign the train data the same data as the temp data
                self.train_data = temp_train.copy(deep=True)
                self.unknown_col = temp_unknown_col.copy(deep=True)
            
            
            curr_iter += 1
        
        #When the loop breaks, update the neighbors list with the reduced training set
        temp_train_with_unknown = self.train_data.copy(deep=True)
        temp_train_with_unknown["unknown_col"] = self.unknown_col.values
        self.distance_matrix = DistanceMatrix(self.test_data, temp_train_with_unknown, self.contAttr, self.discAttr, len(self.contAttr), len(self.discAttr), self.predictionType, "unknown_col")

        self.neighbors = self.distance_matrix.distanceMatrix
        # baseline_acc = classifierAnalyzer.calc_accuracy(method=method)


class CondensedKNN(KNearestNeighbor):
    def __init__(self, test, train, k, contAttr, discAttr, unknown_col='class', predictionType="classification"):
        super().__init__(test, train, k, contAttr, discAttr, unknown_col, predictionType)


    def train(self, acc_delta=0.01, max_iter=1000, remove_correct=False):

        np.random.seed(0)

        curr_iter = 0

        #Initialize X and Z sets
        x_set = np.ones_like(self.unknown_col)
        z_set = np.zeros_like(x_set)

        #Initialize Z with a random subset of X
        z_set[list(np.random.choice(self.train_data.index, self.unknown_col.nunique(), replace=False))] = 1
        x_set[z_set==1] = 0
        
        prev_z_len = 0

        # 3 Exit conditions: Performance is worse than when we started, performance hasn't increased, hit iteration limit
        while prev_z_len * 1.1 < z_set.sum() and curr_iter < max_iter:
            prev_z_len = z_set.sum()
            x_df = self.train_data.iloc[x_set==1]
            z_df = self.train_data.iloc[z_set==1]
            z_unknown = self.unknown_col.iloc[z_set==1]
            testing_KNN = KNearestNeighbor(x_df, z_df, 2, self.contAttr, self.discAttr, unknown_col=z_unknown, predictionType=self.predictionType)

            #Numpy array containing points in the X set classified correctly
            correct_points = (testing_KNN.test()==self.unknown_col[x_set==1])

            tested_indexes = self.train_data.index[x_set==1]
            print("Z size:", z_set.sum())
            for index in tested_indexes[correct_points==False]:
                z_set[index] = 1
                x_set[index] = 0

            #Break if either set is empty
            if x_set.sum() * z_set.sum() == 0: break

           
            
            
            curr_iter += 1

        # 3 Exit conditions: Performance is worse than when we started, performance hasn't increased, hit iteration limit
        self.train_data = self.train_data.drop(self.train_data.index[x_set==1])
        self.unknown_col = self.unknown_col.iloc[z_set==1]
        temp_train_with_unknown = self.train_data.copy(deep=True)
        temp_train_with_unknown["unknown_col"] = self.unknown_col.values
        self.distance_matrix = DistanceMatrix(self.test_data, temp_train_with_unknown, self.contAttr, self.discAttr, len(self.contAttr), len(self.discAttr), self.predictionType, "unknown_col")

        self.neighbors = self.distance_matrix.distanceMatrix
        print(curr_iter)
            

if __name__ == "__main__":
    dataRetriever = DataRetriever("../Datasets/metadata.json")
    dataRetriever.retrieveData("imageSegmentation")
    data = dataRetriever.getDataSet()
    data = data.dropna()
    data = data.reset_index(drop=True)
    # data = data.drop('idNumber', axis=1)

    class_col = dataRetriever.getDataClass()
    # data[class_col] = np.log(data[class_col] + 0.001)
    contAttr = dataRetriever.getContinuousAttributes()
    discAttr = dataRetriever.getDescreteAttributes()

    test = data.sample(frac=0.2, random_state=17)
    train = data.drop(test.index)
    
    # sn = RangeNormalizer(train[contAttr])
    # train[contAttr] = sn.train_fit()


    validate = test.sample(frac=0.5, random_state=13)
    test = test.drop(validate.index)

    train = train.reset_index(drop=True)
    test  = test.reset_index(drop=True)
    validate = validate.reset_index(drop=True)

    # test[contAttr] = sn.fit(test[contAttr])
    # validate[contAttr] = sn.fit(validate[contAttr])

    

    # KNN = KNearestNeighbor(test.drop(class_col, axis=1), train, 5, contAttr, discAttr, unknown_col=class_col, predictionType="regression")
    
    # res = KNN.test() - test[class_col]
    # mean_res = test[class_col].mean() - test[class_col]

    # plt.hist(res, label="Residuals", alpha=0.5, bins=20)
    # plt.hist(mean_res, label="Mean Residuals", alpha=0.5, bins=20)
    # plt.legend()
    # plt.show()
    # test, train, k, contAttr, discAttr, unknown_col='class', predictionType="classification"):
    CKNN = CondensedKNN(test.drop(class_col, axis=1), train, [5], contAttr, discAttr, unknown_col=class_col, predictionType="classification")
    CKNN.train()

    EKNN = EditedKNN(test.drop(class_col, axis=1), validate, train, 5, contAttr, discAttr, unknown_col=class_col, val_unknown_col=class_col, predictionType="classification")
    EKNN.train()

    print("CKNN: ", CKNN.train_data.shape)
    print("EKNN: ", EKNN.train_data.shape)
    