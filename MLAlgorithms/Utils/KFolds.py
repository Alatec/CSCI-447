import pandas as pd
import numpy as np


class KFolds:
    
    def __init__(self, data, k, seed=17, stratisfied=False, class_col='class'):
        """Iterates through data present k train/test sets

        Args:
            data (DataFrame): Data to be segmented into folds
            k (int): number of folds
            seed (int, optional): The random seed used to shuffle the data. Defaults to 17.
        """
        
        #Shuffle the dataset and reset the index
        self.data = data
        self.data.reset_index(drop=True, inplace=True)
        self.data_index = self.data.index.to_numpy()[:]
        self.data_length = len(data)
        self.k = k
        self.mod_val = self.data_length%self.k
        self.fold_size = self.data_length//self.k
        self.stratisfied = stratisfied
        

        if self.stratisfied:
            self.unique_classes = self.data[class_col].unique()

            self.segmented_frames = []
            self.segmented_indexes = []
            self.segmented_mod_vals = []
            self.segmented_fold_size = []

            self.segmented_main_folds = []
            self.segmented_last_folds = []
            for c in self.unique_classes:
                self.segmented_frames.append(self.data[self.data[class_col==c]].reset_index(drop=True))
                self.segmented_indexes.append(self.segmented_frames[-1].index)
                self.segmented_mod_val.append(len(self.segmented_frames[-1])%self.k)
                self.segmented_fold_size.append(len(self.segmented_frames[-1])//self.k)

                self.segmented_main_folds.append([np.arange(self.segmented_fold_size[-1]),np.arange(self.segmented_fold_size[-1],len(self.segmented_frames[-1]))])
                self.segmented_last_folds.append([np.arange(self.segmented_fold_size[-1] + self.segmented_mod_vals[-1]),
                                                    np.arange(self.segmented_fold_size[-1] + self.segmented_mod_vals[-1],len(self.segmented_frames[-1]))])

            


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

        if self.stratisfied:
            #We've gone through all of the folds
            if self.iteration_num == self.k: raise StopIteration

            #First fold shouldn't be rolled
            if self.iteration_num != 0:
                for i,c in enumerate(self.unique_classes):
                    self.segmented_indexes[i] = np.roll(self.segmented_indexes[i], self.segmented_fold_size[i])
                    self.segmented_frames[i].set_index(self.segmented_indexes[i], inplace=True, drop=True, verify_integrity=True)
                
            
        

            #last fold needs to be len(data)%k longer
            if self.iteration_num == self.k-1:
                #Roll k, return k+ len(data)%k
                self.iteration_num += 1
                staging_df_test = []
                staging_df_train = []
                for i,c in enumerate(self.unique_classes):
                    staging_df_test.append(self.segmented_frames[i].loc[self.segmented_last_folds[i][0]])
                    staging_df_train.append(self.segmented_frames[i].loc[self.segmented_last_folds[i][1]])
                
                staging_df_test = pd.concat(staging_df_test, ignore_index=True)
                staging_df_train = pd.concat(staging_df_train, ignore_index=True)

                staging_df_test = staging_df_test.sample(frac=1.0).reset_index(drop=True)
                staging_df_train = staging_df_train.sample(frac=1.0).reset_index(drop=True)
                return (staging_df_test, staging_df_train)
            else:
                self.iteration_num += 1

                staging_df_test = []
                staging_df_train = []
                for i,c in enumerate(self.unique_classes):
                    staging_df_test.append(self.segmented_frames[i].loc[self.segmented_main_folds[i][0]])
                    staging_df_train.append(self.segmented_frames[i].loc[self.segmented_main_folds[i][1]])
                
                staging_df_test = pd.concat(staging_df_test, ignore_index=True)
                staging_df_train = pd.concat(staging_df_train, ignore_index=True)

                staging_df_test = staging_df_test.sample(frac=1.0).reset_index(drop=True)
                staging_df_train = staging_df_train.sample(frac=1.0).reset_index(drop=True)
                return (staging_df_test, staging_df_train)

        else:

            #We've gone through all of the folds
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
        
        