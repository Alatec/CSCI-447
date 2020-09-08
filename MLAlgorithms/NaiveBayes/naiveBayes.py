import pandas as pd
class NaiveBayes():


    def __init__(self, dataFrame, dataClass):
        self.unknownVal = "unk"

        self.dataFrame = dataFrame
        self.dataClass = dataClass

        self.separatedClasses = self.seperateDataByClass()
        self.classPriors = self.calculateClassPriors()
        self.d = len(next(iter(self.separatedClasses.values())).columns)

        self.trainCalculation = self.train()

    # def retrain(self, dataFrame, dataClass):
    #     self.unknownVal = "unk"
    #
    #     self.dataFrame = dataFrame
    #     self.dataClass = dataClass
    #
    #     self.separatedClasses = self.seperateDataByClass()
    #     self.classPriors = self.calculateClassPriors()
    #     self.d = len(next(iter(self.separatedClasses.values())).columns)
    #
    #     self.trainCalculation = self.train()

    def test(self, testFrame):
        classProbs = {}
        results = []
        for test in testFrame.iterrows():
            for dataClass in self.trainCalculation.keys():
                classProb = 1
                for feature in self.trainCalculation[dataClass].keys():
                    if test[1][feature] in self.trainCalculation[dataClass][feature]:
                        classProb *= self.trainCalculation[dataClass][feature][test[1][feature]]
                    else:
                        classProb *= self.trainCalculation[dataClass][feature][self.unknownVal]
                classProbs[dataClass] = self.classPriors[dataClass] * classProb
            results.append(max(classProbs, key=classProbs.get))
        return results

    # This method trains the algorithm and returns a tuple of a dictionary containing trained values
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

        self.separatedClasses = separatedClasses
        return separatedClasses
