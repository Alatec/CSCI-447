import numpy as np


class OneHotEncoder():
    
    def __init__(self):
        """
        docstring
        """
        self.encodedDict = {}
    
    def train_fit(self, dataFrame, categoricalData):
        """
        Encodes the categorical data with the One-Hot Encoding method

        Args:
            dataFrame: Pandas DataFrame
            categoricalData: NumpyArray<String>

        Returns:
            dataFrameEncoded: Pandas DataFrame
        """
        if len(categoricalData) < 1:
            return dataFrame

        dataFrameEncoded = dataFrame.copy()


        # Will encode each discrete attribute
        for attr in categoricalData:
            self.encodedDict[attr] = []
            allVals = dataFrameEncoded[attr].unique()
            allValsLen = len(allVals)

            if allValsLen > 2:
                one_hot_cols = np.zeros((len(dataFrameEncoded), allValsLen+1))
                for i, val in enumerate(allVals):
                    self.encodedDict[attr].append((val, f'{attr}_{val}'))
                    one_hot_cols[dataFrameEncoded[attr]==val,i] = 1
                    dataFrameEncoded[self.encodedDict[attr][i][1]] = one_hot_cols[:,i]
                self.encodedDict[attr].append((f'{attr}_catchall',))
                dataFrameEncoded[self.encodedDict[attr][-1][0]] = one_hot_cols[:,-1]
                dataFrameEncoded = dataFrameEncoded.drop(attr, axis=1)
            else:
                one_hot_col = np.zeros(len(dataFrameEncoded), dtype=np.float64)
                self.encodedDict[attr].append((allVals[0], allVals[1]))
                one_hot_col[dataFrameEncoded[attr]==allVals[0]] = 1
                dataFrameEncoded = dataFrameEncoded.drop(attr, axis=1)
                dataFrameEncoded[attr] = one_hot_col

        return dataFrameEncoded

    def fit(self, dataFrame):
        
        for attr, value in self.encodedDict.items():
            
            if len(value) > 1:
                one_hot_cols = np.zeros((len(dataFrame), len(value)))
                col_order = []
                catchall_col = None
                for i, val in enumerate(value):
                    if len(val) == 1: 
                        catchall_col = val[0]
                    else:
                        col_order.append(val[1])
                        one_hot_cols[:, len(col_order)-1] = (dataFrame[attr]==val[0])
                        dataFrame[col_order[-1]] = one_hot_cols[:,len(col_order)-1]
                
                one_hot_cols[:,-1] = one_hot_cols.sum(axis=1)
                one_hot_cols[:,-1] = np.logical_not(one_hot_cols[:,-1] > 0)
                dataFrame[catchall_col] = one_hot_cols[:,-1]
                dataFrame = dataFrame.drop(attr, axis=1)
            else:
                one_hot_col = np.zeros(len(dataFrame), dtype=np.float64)
                val = self.encodedDict[attr][0]
                one_hot_col[dataFrame[attr]==val[0]] = 1
                dataFrame = dataFrame.drop(attr, axis=1)
                dataFrame[attr] = one_hot_col
        return dataFrame



    



            

        
        



## OG Column name
##      New Column Names OG_red OG_green
##          Associtiated Values red green

## OG_red[OG==red] = 1

## OG = red, green, green, red
