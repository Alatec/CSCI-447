import pandas as pd

class NaiveBayes():


    def __init__(self, dataFrame, dataClass):
        self.dataFrame = dataFrame
        self.dataClass = dataClass

        self.separatedClasses = self.seperateDataByClass()
        self.qCalculation = self.calculateQ()



    def calculateQ(self):
        qDict = {}

        for dataClass, data in self.separatedClasses.items():
            qDict[dataClass] = {}
            for feature in data:
                qDict[dataClass][feature] = {}
                for value in data[feature].unique():
                    qDict[dataClass][feature][value] = {}
                    qDict[dataClass][feature][value] = len(data[data[feature] == value]) / len(data)

        return qDict


    # Separates the data set into a dictionary with the key being the class
    # and the value being a dataframe of rows with that class
    def seperateDataByClass(self):
        separatedClasses = {}

        for i in self.dataFrame[self.dataClass].unique():
            separatedClasses[i] = self.dataFrame[self.dataFrame[self.dataClass] == i]

        for i in separatedClasses.keys():
            separatedClasses[i] = separatedClasses[i].drop(columns=self.dataClass)

        self.separatedClasses = separatedClasses
        return separatedClasses



