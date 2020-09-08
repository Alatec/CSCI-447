import pandas as pd
class NaiveBayes():


    def __init__(self, dataFrame, dataClass):
        self.unknownVal = "arbitraryValue"

        self.dataFrame = dataFrame
        self.dataClass = dataClass

        self.separatedClasses = self.seperateDataByClass()
        self.classPriors = self.calculateClassPriors()
        self.d = len(next(iter(self.separatedClasses.values())).columns)

        self.traindCalculation = self.train()

    # This method tests the trained algorithm with a given pandas test data frame.
    def test(self, testFrame):
        classProbs = {}
        results = []

        for test in testFrame.iterrows():
            for dataClass in self.traindCalculation.keys():
                classProb = 1
                for feature in self.traindCalculation[dataClass].keys():
                    if test[1][feature] in self.traindCalculation[dataClass][feature]:
                        classProb *= self.traindCalculation[dataClass][feature][test[1][feature]]
                    else:
                        classProb *= self.traindCalculation[dataClass][feature][self.unknownVal]
                classProbs[dataClass] = self.classPriors[dataClass] * classProb
            results.append(max(classProbs, key=classProbs.get))
        return results

    # This method trains the algorithm and returns a nested dictionary of value probabilities
    def train(self):
        trainDict = {}

        for dataClass, data in self.separatedClasses.items():
            trainDict[dataClass] = {}

            for feature in data:

                trainDict[dataClass][feature] = {}
                trainDict[dataClass][feature][self.unknownVal] = (1 / len(data)) + self.d
                for value in data[feature].unique():
                    numerator = len(data[data[feature] == value]) + 1
                    denominator = len(data) + self.d

                    trainDict[dataClass][feature][value] = numerator / denominator

        return trainDict

    # Calculates the ClassPriors
    def calculateClassPriors(self):
        classPriors = {}

        for dataClass, data in self.separatedClasses.items():
            classPriors[dataClass] = len(data) / len(self.dataFrame)

        return classPriors

    # This method returns a dictionary where the keys are classes and the values
    # are dataframes of classes for their respective key
    def seperateDataByClass(self):
        separatedClasses = {}

        for i in self.dataFrame[self.dataClass].unique():
            separatedClasses[i] = self.dataFrame[self.dataFrame[self.dataClass] == i]

        for i in separatedClasses.keys():
            separatedClasses[i] = separatedClasses[i].drop(columns=self.dataClass)

        return separatedClasses
