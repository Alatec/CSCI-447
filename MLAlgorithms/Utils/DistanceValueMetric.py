import numpy as np
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
    def __init__(self, dataFrame, classifier, discreteValues):
        self.matrixDict = {}

        for attr in discreteValues:
            self.matrixDict[attr] = self._createMatrix(dataFrame, classifier, attr)
            break

    def calculateDistance(self, x, y):
        pass

    def _createMatrix(self, dataFrame, classifier, attr):
        uniqueValues = dataFrame[attr].unique()
        uniqueValuesLen = len(uniqueValues)

        matrix = np.zeros(shape=(uniqueValuesLen, uniqueValuesLen), dtype='string_')

        for row in range(0, len(matrix)):
            for col in range(0, len(matrix[row])):
                print(f"{uniqueValues[col]} + {uniqueValues[row]}")
                matrix[row][col] = f"{uniqueValues[col]} + {uniqueValues[row]}"

            # If the data set is regression, we need to bin it into classes



        print(uniqueValues)
        print(matrix)

        return None
