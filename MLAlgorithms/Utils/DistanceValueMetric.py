import numpy as np
import pandas as pd

class DistanceValueMetric:

    """
    On instantiation, a dictionary of Value Difference Metrics are created for each attribute in a data set

    Args:
        dataFrame: Pandas DataFrame
        classifier: String
        discreteValues: List<String>
    """
    def __init__(self, dataFrame, classifier, discreteValues, predictionType):
        self.matrixDict = {}
        self.wonskianMetric = 1

        discreteValuesSorted = sorted(discreteValues)
        if classifier in discreteValuesSorted:
            discreteValuesSorted.remove(classifier)

        for attr in discreteValuesSorted:
            self.matrixDict[attr] = self._createMatrix(dataFrame, classifier, attr, predictionType)

    """
    Pass in two numpy array representing data points and the attribute dictionary to define indexes to calclate
    the distance between categorical values
    
    Args:
        x: Array<Object>
        y: Array<Object>
        attrDict: Dict<Object, int>
        
    Returns:
        sum: int
    """
    def calculateDistance(self, x, y, attrDict):
        sum = 0
        for attr in self.matrixDict.keys():
            sum += self.matrixDict[attr][x[attrDict[attr]]][y[attrDict[attr]]]
            # sum += self.matrixDict[attr][x[attr]][y[attr]]

        sum = sum**(1/self.wonskianMetric)

        return sum

    """
    This is a private function that will create the pandas data frame that holds the Value Difference Metric Matrix
    
    Still need to be able to handle regression data
    """
    def _createMatrix(self, dataFrame, classifier, attr, predictionType):
        uniqueValues = sorted(dataFrame[attr].unique())
        uniqueValuesLen = len(uniqueValues)
        uniqueValuesCount = {}
        uniqueValuesCountWithClass = {}

        if predictionType == "classification":
            uniqueClassifiers = sorted(dataFrame[classifier].unique())
        else:
            uniqueClassifiers = np.histogram(dataFrame[classifier], bins=8)
            uniqueClassifiers = uniqueClassifiers[0]


        # Create a dictionary of value counts both in total and in class specific total
        # This is use for the delta calculation
        for val in uniqueValues:
            uniqueValuesCount[val] = len(dataFrame[dataFrame[attr] == val])
            uniqueValuesCountWithClass[val] = {}
            for clss in uniqueClassifiers:
                uniqueValuesCountWithClass[val][clss] = len(dataFrame[(dataFrame[attr] == val)
                                                                      & (dataFrame[classifier] == clss)])

        # Create the the matrix that holds the delta value for each value combination
        numpyMatrix = np.zeros(shape=(uniqueValuesLen, uniqueValuesLen))
        for row in range(0, len(numpyMatrix)):
            for col in range(0, len(numpyMatrix[row])):

            # Calculate Delta for each element in the matrix
                delta = 0
                for clss in uniqueClassifiers:
                    delta += abs((uniqueValuesCountWithClass[uniqueValues[col]][clss]
                                  / uniqueValuesCount[uniqueValues[col]])
                                 - (uniqueValuesCountWithClass[uniqueValues[row]][clss]
                                    / uniqueValuesCount[uniqueValues[row]]))**self.wonskianMetric

                numpyMatrix[col][row] = delta

        matrix = pd.DataFrame(numpyMatrix, uniqueValues, columns=uniqueValues)

        return matrix

    def lookUp(self, attr, val):
        print(self.matrixDict[attr])
        return self.matrixDict[attr][val]

    def printMatrix(self):
        for attr in self.matrixDict.keys():
            print(self.matrixDict[attr])

    def printSpecificMatrix(self, key):
        print(self.matrixDict[key])

