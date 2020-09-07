import pandas as pd

class NaiveBayes():


    def __init__(self):
        pass



def calculateQ(dataFrame, classifier):
    pass

def seperateDataByClass(dataFrame, classifier):
    seperatedClassifiers = {}

    for i in dataFrame.iterrows():
        if classifier not in seperatedClassifiers.keys():
            seperatedClassifiers[classifier] = []
        seperatedClassifiers[classifier].append(i[1][:-1]) # Only works if the classifier is the last element

    return seperatedClassifiers


