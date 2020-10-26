import numpy as np


class OneHotEncoder():
    
    def __init__(self):
        """
        docstring
        """
        self.encodedDict = None
    
    def train(self):
        pass

    def fit(self, data):
        pass
    """
    Encodes the categorical data with the One-Hot Encoding method

    Args:
        dataFrame: Pandas DataFrame
        categoricalData: NumpyArray<String>

    Returns:
        dataFrameEncoded: Pandas DataFrame
    """
    def oneHotEncoder(self, dataFrame, categoricalData):
        if len(categoricalData) < 1:
            return dataFrame

        dataFrameEncoded = dataFrame.copy()


        # Will encode each discrete attribute
        for attr in categoricalData:
            attrDict = {}
            allVals = dataFrameEncoded[attr].unique()
            allValsLen = len(allVals)

            # Integer encoding
            for i, val in enumerate(allVals):
                array = np.zeros(allValsLen, dtype=int)
                array[i] = 1
                attrDict[val] = array

            dataFrameEncoded[attr] = self._encoderHelper(dataFrameEncoded[attr], attrDict)

        self.encodedDict = attrDict
        return dataFrameEncoded


    def _encoderHelper(self, data, attrDict):
        data = data.apply(lambda x: attrDict[x])

        return data


## OG Column name
##      New Column Names OG_red OG_green
##          Associtiated Values red green

## OG_red[OG==red] = 1

## OG = red, green, green, red
