import pandas as pd

class NaiveBayes():


    def __init__(self, dataFrame, dataClass):
        self.dataFrame = dataFrame
        self.dataClass = dataClass

        self.separatedClasses = self.seperateDataByClass()
        self.d = len(next(iter(self.separatedClasses.values())).columns)

        self.qCalculation, self.tbnCalculation = self.train()


    def retrain(self, dataFrame, dataClass):
        self.dataFrame = dataFrame
        self.dataClass = dataClass

        self.separatedClasses = self.seperateDataByClass()
        self.d = len(next(iter(self.separatedClasses.values())).columns)

        self.qCalculation, self.tbnCalculation = self.train()

    def test(self, testFrame):
        pass

    # This method trains the algorithm and returns a tuple of a dictionary containing qValues and a dictionary
    # containing trained values
    def train(self):
        qDict = {}
        tbnDict = {}

        for dataClass, data in self.separatedClasses.items():
            qDict[dataClass] = {}
            tbnDict[dataClass] = {}
            for feature in data:
                qDict[dataClass][feature] = {}
                tbnDict[dataClass][feature] = {}
                for value in data[feature].unique():

                    qnumerator = len(data[data[feature] == value])
                    qdenominator = len(data)
                    qValue = qnumerator / qdenominator
                    qDict[dataClass][feature][value] = qValue

                    tbnDict[dataClass][feature][value] = (qnumerator + 1) / (qdenominator + self.d)



        return (qDict, tbnDict)


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




