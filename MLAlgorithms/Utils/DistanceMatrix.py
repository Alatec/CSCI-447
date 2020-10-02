from MLAlgorithms.Utils.DistanceValueMetric import DistanceValueMetric
from MLAlgorithms.Utils.ValueDistanceMetric import ValueDifferenceMetric
from MLAlgorithms.Utils.NumbaFunctions import calculate_euclid_distances
from tqdm import tqdm
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)

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

    def _createContMatrix(self, testSet, trainSet, contAttr):
        result = calculate_euclid_distances(testSet[contAttr].to_numpy(dtype=np.float64), trainSet[contAttr].to_numpy(dtype=np.float64))
        return result

    def _createDiscMatrix(self, testSet, trainSet, classifier, discAttr, predictionType):
        dataMetric = ValueDifferenceMetric(testSet[list(set(discAttr + [classifier]))], unknown_col=classifier, prediction_type=predictionType)
        dataMetric.train()
        # dataMetric.calc_distance_matrix(testSet[set(discAttr)], trainSet[set(discAttr)])
        # dataMetric = DistanceValueMetric(testSet, classifier, discAttr, predictionType)
        return dataMetric

    def _createDistanceMatrix(self, contMatrix, discMatrix, testSet, trainSet, trainLen, testLen, alpha, beta, classifier, discAttr):

        distanceMatrix = np.zeros((testLen, trainLen))


        discMatrix.calc_distance_matrix(testSet[discAttr], trainSet[discAttr])
        distanceMatrix += alpha*contMatrix
        distanceMatrix += beta*discMatrix.distances

        # something = pd.DataFrame(distanceMatrix)

        return distanceMatrix

    def recalculateCentriods(self, centroids):
        self.contMatrix = self._createContMatrix(self.testSet, centroids, self.contAttr)
        self.discMatrix = self._createDiscMatrix(self.testSet, centroids, self.classifier, self.discAttr, self.predictionType)
        self.distanceMatrix = self._createDistanceMatrix(self.contMatrix, self.discMatrix, self.testSet, centroids, len(self.trainSet), len(self.testSet), self.alpha, self.beta, self.classifier, self.discAttr)
        # print(self.distanceMatrix)

