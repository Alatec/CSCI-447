from MLAlgorithms.Utils.DistanceValueMetric import DistanceValueMetric
from MLAlgorithms.Utils.NumbaFunctions import calculate_euclid_distances
import numpy as np

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

        self.contMatrix = self._createContMatrix(testSet, contAttr)
        self.discMatrix = self._createDiscMatrix(self.trainSet, self.classifier, self.discAttr, self.predictionType)

        self.distanceMatrix = self._createDistanceMatrix(self.contMatrix, self.discMatrix)

    def _createContMatrix(self, testSet, trainSet, contAttr):
        result = calculate_euclid_distances(testSet[contAttr].to_numpy(dtype=np.float64), trainSet[contAttr].to_numpy(dtype=np.float64))
        return result

    def _createDiscMatrix(self, trainSet, classifier, discAttr, predictionType):
        dataMetric = DistanceValueMetric(trainSet, classifier, discAttr, predictionType)
        return dataMetric

    def _createDistanceMatrix(self, contMatrix, discMatrix):
        return None

