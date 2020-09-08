import pandas as pd
import numpy as np


class KFolds:
    
    def __init__(self, data, k, seed=17):
        """Iterates through data present k train/test sets

        Args:
            data (DataFrame): Data to be segmented into folds
            k (int): number of folds
            seed (int, optional): The random seed used to shuffle the data. Defaults to 17.
        """

        #Shuffle the dataset and reset the index
        self.data = data.sample(frac=1.0, random_state=seed)
        self.data.reset_index(drop=True, inplace=True)
        self.data_index = self.data.index.to_numpy()[:]
        self.data_length = len(data)
        self.k = k
        self.mod_val = self.data_length%self.k
        self.fold_size = self.data_length//self.k

        #We need to pass in a list of rows indices to use for the test, train sets
        #Since the indices are the same every time, we make them here
        self.main_folds = [np.arange(self.fold_size),np.arange(self.fold_size,self.data_length)]
        self.last_folds = [np.arange(self.fold_size+self.mod_val), np.arange(self.fold_size+self.mod_val,self.data_length)]
        
    def __len__(self):
        return self.k

    def __iter__(self):
        self.iteration_num = 0
        return self

    def __next__(self):

        #We've go through all of the folds
        if self.iteration_num == self.k: raise StopIteration

        #First fold shouldn't be rolled
        if self.iteration_num != 0:
            self.data_index = np.roll(self.data_index, self.fold_size)
            self.data.set_index(self.data_index, inplace=True, drop=True, verify_integrity=True)
            
        
    

        #last fold needs to be len(data)%k longer
        if self.iteration_num == self.k-1:
            #Roll k, return k+ len(data)%k
            self.iteration_num += 1
            return (self.data.loc[self.last_folds[0]], self.data.loc[self.last_folds[1]])
        else:
            self.iteration_num += 1
            return (self.data.loc[self.main_folds[0]], self.data.loc[self.main_folds[1]])
        
        