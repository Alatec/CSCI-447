import numpy as np
import pandas as pd

from MLAlgorithms.Errors.UtilError import UndefinedPerformaceMethod

class ClassifierAnalyzer:
    
    def __init__(self, truth_labels, pred_labels, positive_label='1', negative_label='0'):
        self.truth_labels = truth_labels
        self.pred_labels = np.asarray(pred_labels)

        self.unique_labels = list(set(self.truth_labels))
        self.multi_class = False
        self.positive_label = positive_label
        self.negative_label = negative_label
        # if len(self.unique_labels)>2: self.multi_class = True
        self.multi_class = True
        rows_list = []
        for label in self.unique_labels:
            row_dict = {}
            row_dict['Truth'] = str(label)
            for pred in self.unique_labels:
                row_dict[str(pred)] = ((self.truth_labels==label)&(self.pred_labels==pred)).sum()

            # left_overs = set(self.unique_labels) - set(row_dict.keys())
            #
            # for left in left_overs:
            #     row_dict[str(left)] = 0
            rows_list.append(row_dict)
        
        self.confusion_matrix = pd.DataFrame(rows_list)
        self.confusion_matrix = self.confusion_matrix.set_index('Truth', drop=True)

        # for col in self.confusion_matrix.index:
        #     if col not in self.confusion_matrix.columns:
        #         self.confusion_matrix[col] = pd.Series(np.zeros(len(self.confusion_matrix), dtype=np.int64), index=self.confusion_matrix.index)

    def calc_accuracy(self, method=None):
        if self.multi_class:
            if method is None: raise UndefinedPerformaceMethod(True)
            elif method == "macro":
                average = 0
                for label in self.confusion_matrix.index:
                    temp = self.confusion_matrix.loc[label, label]

                    temp += self.confusion_matrix.loc[self.confusion_matrix.index != label, self.confusion_matrix.columns != label].sum().sum()

                    average += temp/self.confusion_matrix.sum().sum()

                return average/(len(self.confusion_matrix))
            elif method == "micro":
                num = 0
                for label in self.confusion_matrix.index:
                    num += self.confusion_matrix.loc[label,label]
                return num/self.confusion_matrix.sum().sum()
            
        else:
            # if method is not None: raise UndefinedPerformaceMethod
            t = self.positive_label
            n = self.negative_label
            return (self.confusion_matrix.loc[t,t]+self.confusion_matrix.loc[n,n])/self.confusion_matrix.sum().sum()

    def calc_precision(self, method=None):
        if self.multi_class:
            if method is None: raise UndefinedPerformaceMethod(True)
            elif method == "macro":
                average = 0.0
                for label in self.confusion_matrix.index:
                    temp = self.confusion_matrix.loc[label, label].sum()
                    dem = self.confusion_matrix.loc[:, self.confusion_matrix.columns == label].sum().sum()
                    if dem == 0:
                        temp = 0
                    else:
                        temp /= self.confusion_matrix.loc[:, self.confusion_matrix.columns == label].sum().sum()
                    average += temp

                  
                return average/(len(self.confusion_matrix))
            elif method == "micro":
                return self.calc_accuracy(method=method)
            
        else:
            if method is not None: raise UndefinedPerformaceMethod

            t = self.positive_label
            n = self.negative_label
            return (self.confusion_matrix.loc[t,t])/self.confusion_matrix.loc[:,t].sum()

    def calc_recall(self, method=None):
        if self.multi_class:
            if method is None: raise UndefinedPerformaceMethod(True)
            elif method == "macro":
                average = 0
                for label in self.confusion_matrix.index:
                    temp = self.confusion_matrix.loc[label, label].sum()
                    dem = self.confusion_matrix.loc[self.confusion_matrix.columns == label].sum().sum()
                    if dem == 0:
                        temp = 0
                    else:
                        temp /= self.confusion_matrix.loc[self.confusion_matrix.columns == label].sum().sum()
                    average += temp

                return average/(len(self.confusion_matrix))
            elif method == "micro":
                return self.calc_accuracy(method=method)
            
        else:
            if method is not None: raise UndefinedPerformaceMethod

            t = self.positive_label
            n = self.negative_label
            return (self.confusion_matrix.loc[t,t])/self.confusion_matrix.loc[t].sum()

    def calc_f1_score(self, method=None):
        prec = self.calc_precision(method=method)
        recal = self.calc_recall(method=method)
        sum_val = prec+recal
        if sum_val == 0: return 0
        else: return 2*prec*recal/(sum_val)