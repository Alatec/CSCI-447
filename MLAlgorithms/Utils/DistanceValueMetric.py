import numpy as np
import pandas as pd
"""
for each feature in data set
    var 1 = count how many values each feature has                    For instance, for the feature type color, how many blues, reds, ect exist
    var 2 = count how many values each feature has and where the class is equal to some variable a

    Construct the feature difference matrix
"""

"""
asd

Args:
    dataFrame: Pandas DataFrame
    
Returns:
    

"""
class DistanceValueMetric:

    """
    Args:
        dataFrame: Pandas DataFrame
        classifier: String
        discreteValues: List<String>
    """
    def __init__(self, dataFrame, classifier, discreteValues, predictionType):
        self.matrixDict = {}
        self.wonskianMetric = 1

        discreteValuesSorted = sorted(discreteValues)
        discreteValuesSorted.remove(classifier)

        for attr in discreteValuesSorted:
            self.matrixDict[attr] = self._createMatrix(dataFrame, classifier, attr, predictionType)


    """
    Pass in two data points to calculate the the total distance of categorical data
    """
    def calculateDistance(self, x, y):
        sum = 0
        for attr in self.matrixDict.keys():
            sum += self.matrixDict[attr][x[attr].item()][y[attr].item()]

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
        uniqueClassifiers = sorted(dataFrame[classifier].unique())


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

    def printMatrix(self):
        for attr in self.matrixDict.keys():
            print(self.matrixDict[attr])
            print(self.matrixDict[attr]["?"])
            break

