import numpy as np

"""
Encodes the categorical data with the One-Hot Encoding method

Args:
    dataFrame: Pandas DataFrame
    categoricalData: NumpyArray<String>

Returns:
    dataFrameEncoded: Pandas DataFrame
"""
def oneHotEncoder(dataFrame, categoricalData):
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

        dataFrameEncoded[attr] = _encoderHelper(dataFrameEncoded[attr], attrDict)

    return dataFrameEncoded


def _encoderHelper(data, attrDict):

    data = data.apply(lambda x: attrDict[x])

    return data
