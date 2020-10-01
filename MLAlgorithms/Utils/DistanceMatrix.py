from MLAlgorithms.Utils.DistanceValueMetric import DistanceValueMetric
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

        self.discMatrix = self._createDiscMatrix(self.testSet, self.classifier, self.discAttr, self.predictionType)
        self.contMatrix = self._createContMatrix(self.testSet, self.trainSet, self.contAttr)

        self.distanceMatrix = self._createDistanceMatrix(self.contMatrix, self.discMatrix, testSet, trainSet, len(trainSet), len(testSet), alpha, beta)

    def _createContMatrix(self, testSet, trainSet, contAttr):
        result = calculate_euclid_distances(testSet[contAttr].to_numpy(dtype=np.float64), trainSet[contAttr].to_numpy(dtype=np.float64))
        return result

    def _createDiscMatrix(self, testSet, classifier, discAttr, predictionType):
        dataMetric = DistanceValueMetric(testSet, classifier, discAttr, predictionType)
        return dataMetric

    def _createDistanceMatrix(self, contMatrix, discMatrix, testSet, trainSet, trainLen, testLen, alpha, beta):

        distanceMatrix = np.zeros((testLen, trainLen))
        for x in range(0, len(distanceMatrix)):
            for y in range(0, len(distanceMatrix[1])):
                distanceMatrix[x,y] += alpha*contMatrix[x, y]
                distanceMatrix[x,y] += beta*discMatrix.calculateDistance(testSet.iloc[[x]], trainSet.iloc[[y]])

        # something = pd.DataFrame(distanceMatrix)

        return distanceMatrix

    def recalculateCentriods(self, centroids):
        self.contMatrix = self._createContMatrix(self.testSet, centroids, self.contAttr)
        self.distanceMatrix = self._createDistanceMatrix(self.contMatrix, self.discMatrix, self.testSet, centroids, len(self.trainSet), len(self.testSet), self.alpha, self.beta)
        # print(self.distanceMatrix)

