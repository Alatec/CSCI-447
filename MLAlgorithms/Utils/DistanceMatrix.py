from MLAlgorithms.Utils.ValueDifferenceMetric import ValueDifferenceMetric
from MLAlgorithms.Utils.NumbaFunctions import calculate_euclid_distances
import numpy as np
np.set_printoptions(threshold=np.inf)


"""
This class create a 2D matrix of scalar values representing the distance between two rows of data points

Args:
    testSet: Pandas DataFrame
    trainSet: Pandas DataFrame
    conAttr: List<String>
    disAttr: List<String>
    alpha: float
    beta: float
    predictionType: String
    classifier: String

"""
class DistanceMatrix():
    def __init__(self, testSet, trainSet, contAttr, discAttr, alpha, beta, predictionType, classifier):
        self.testSet = testSet
        self.trainSet = trainSet
        self.contAttr = contAttr
        self.discAttr = discAttr
        self.alpha = alpha
        self.beta = beta
        self.predictionType = predictionType
        self.classifier = classifier

        if self.classifier in self.discAttr:
            self.discAttr.remove(classifier)

        if self.classifier in self.contAttr:
            self.contAttr.remove(classifier)

        self.discMatrix = self._createDiscMatrix(self.testSet, self.trainSet, self.classifier, self.discAttr, self.predictionType)
        self.contMatrix = self._createContMatrix(self.testSet, self.trainSet, self.contAttr)

        self.distanceMatrix = self._createDistanceMatrix(self.contMatrix, self.discMatrix, testSet, trainSet, len(trainSet), len(testSet), alpha, beta, self.classifier, self.discAttr)

    """
    Private method to create the continuous distance matrix
        
    Args:
        testSet: Pandas DataFrame
        trainSet: Pandas DataFrame
        contAttr: List<String>
            
    """
    def _createContMatrix(self, testSet, trainSet, contAttr):
        result = calculate_euclid_distances(testSet[contAttr].to_numpy(dtype=np.float64), trainSet[contAttr].to_numpy(dtype=np.float64))
        return result

    """
    Private method to create the categorical distance matrix

    Args:
        testSet: Pandas DataFrame
        trainSet: Pandas DataFrame
        classifier: String
        discAttr: List<String>
        predictionType: String

    """
    def _createDiscMatrix(self, testSet, trainSet, classifier, discAttr, predictionType):
        dataMetric = ValueDifferenceMetric(trainSet[list(set(discAttr + [classifier]))], unknown_col=classifier, prediction_type=predictionType)
        dataMetric.train()
        return dataMetric

    """
    Private method to create the distance matrix

    Args:
        contMatrix: 2D Numpy Array of float
        discMatrix: 2D Numpy Array of float
        testSet: Pandas DataFrame
        trainSet: Pandas DataFrame
        trainLen: int
        testLen: int
        alpha: float
        beta: float
        classifier: String
        discAttr: List<String>

    """
    def _createDistanceMatrix(self, contMatrix, discMatrix, testSet, trainSet, trainLen, testLen, alpha, beta, classifier, discAttr):

        distanceMatrix = np.zeros((testLen, trainLen))


        discMatrix.calc_distance_matrix(testSet[discAttr], trainSet[discAttr])
        distanceMatrix += alpha*contMatrix
        distanceMatrix += beta*discMatrix.distances

        return distanceMatrix

    """
    A method to recalculate the distance between centroids and data points

    Args:
        centroids: Pandas DataFrame

    """
    def recalculateCentriods(self, centroids):
        self.contMatrix = self._createContMatrix(self.testSet, centroids, self.contAttr)
        self.discMatrix = self._createDiscMatrix(self.testSet, centroids, self.classifier, self.discAttr, self.predictionType)
        self.distanceMatrix = self._createDistanceMatrix(self.contMatrix, self.discMatrix, self.testSet, centroids, len(self.trainSet), len(self.testSet), self.alpha, self.beta, self.classifier, self.discAttr)


