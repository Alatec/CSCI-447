import pandas as pd

class NaiveBayes():


    def __init__(self):
        self.idea = None



    def calculateQ(self, dataFrame, classifier):
        pass


    def seperateDataByClass(self, dataFrame, dataClass):
        separatedClasses = {}

        for i in dataFrame[dataClass].unique():
            separatedClasses[i] = dataFrame[dataFrame[dataClass] == i]

        return separatedClasses



