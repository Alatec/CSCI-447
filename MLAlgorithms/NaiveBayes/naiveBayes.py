import pandas as pd
class NaiveBayes():


    def __init__(self, dataFrame, dataClass):
        """Creates an instance of a trained NaiveBayes Algorithm

        Args:
            dataFrame (DataFrame): The dataset that will train the algorithm
            dataClass (String): The class attribute for a given data set
        """
        self.unknownVal = "arbitraryValue"

        self.dataFrame = dataFrame
        self.dataClass = dataClass

        self.separatedClasses = self.seperateDataByClass()
        self.classPriors = self.calculateClassPriors()
        self.d = len(next(iter(self.separatedClasses.values())).columns)

        self.trainedCalculation = self.train()


    def test(self, testFrame):
        """Tests the NaiveBayes instance will a given test data set

        Args:
            testFrame (DataFrame): The test set

        Returns:
            results (List<PredictedValues>): A list of predicted values for the given test set
        """
        classProbs = {}
        results = []

        for test in testFrame.iterrows():
            for dataClass in self.trainedCalculation.keys():
                classProb = 1
                for feature in self.trainedCalculation[dataClass].keys():
                    if test[1][feature] in self.trainedCalculation[dataClass][feature]:
                        classProb *= self.trainedCalculation[dataClass][feature][test[1][feature]]
                    else:
                        classProb *= self.trainedCalculation[dataClass][feature][self.unknownVal]
                classProbs[dataClass] = self.classPriors[dataClass] * classProb
            results.append(max(classProbs, key=classProbs.get))
        return results

    def train(self):
        """On creation of the object, the object will create a trained dictionary based on the input data set"""
        trainDict = {}

        for dataClass, data in self.separatedClasses.items():
            trainDict[dataClass] = {}

            for feature in data:

                trainDict[dataClass][feature] = {}
                trainDict[dataClass][feature][self.unknownVal] = (1 / (len(data) + self.d))
                for value in data[feature].unique():
                    numerator = len(data[data[feature] == value]) + 1
                    denominator = len(data) + self.d

                    trainDict[dataClass][feature][value] = numerator / denominator

        return trainDict

    def calculateClassPriors(self):
        """On creation of the object, the object will calculate the class priors"""

        classPriors = {}

        for dataClass, data in self.separatedClasses.items():
            classPriors[dataClass] = len(data) / len(self.dataFrame)

        return classPriors

    def seperateDataByClass(self):
        """On creation of the object, the object will separate the data set by class.
            Once separated, the class column will be dropped, leaving a 'Class':'DataFrame'
         """
        separatedClasses = {}

        for i in self.dataFrame[self.dataClass].unique():
            separatedClasses[i] = self.dataFrame[self.dataFrame[self.dataClass] == i]

        for i in separatedClasses.keys():
            separatedClasses[i] = separatedClasses[i].drop(columns=self.dataClass)

        return separatedClasses
