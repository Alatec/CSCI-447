import json, os, pandas as pd


class DataRetriever():
    def __init__(self, path):
        self.currentData = None
        self.data = None
        self.dataSetPath = os.path.split(path)[0]
        self.metaDataPath = path
        self.menu = []
        self.dataSet = None
        self.dataClass = None
        self.continuousAttributes = None
        self.rowsToDrop = None

        self._buildDataMenu()

    # Builds a menu from the meta data of a data set
    def _buildDataMenu(self):

        with open(self.metaDataPath) as f:
            self.data = json.load(f)

        for object in self.data:
            self.menu.append(object)

    ############## Getters ##############
    def getContinuousAttributes(self):
        return self.continuousAttributes

    def getDescreteAttributes(self):
        return self.descreteAttributes

    def getDataMenu(self):
        return self.menu

    def getDataClass(self):
        return self.dataClass

    def getDataSet(self):
        self.dataSet = self.dataSet.sample(frac=1, random_state=69)
        return self.dataSet.reset_index(drop=True)

    def getRowsToDrop(self):
        return self.rowsToDrop

    def getPredictionType(self):
        return self.predictionType

    #####################################

    def hasData(self, data):
        return data in self.menu

    # Returns a data frame of the given string
    # Will return null if the data doesn't exist
    def retrieveData(self, data):
        if not self.hasData(data):
            return None

        with open(self.metaDataPath) as f:
            js = json.load(f)
            jsonData = js[data]

        dataPath = self.dataSetPath + "/" + jsonData["dataPath"]
        header = jsonData["attributes"]
        naValues = jsonData["NAValues"]
        dataClass = jsonData['class']
        continuousAttributes = jsonData['continuous']
        discreteAttributes = jsonData['discrete']

        dataSet = pd.read_csv(self.dataSetPath + "/" + dataPath, names=header, na_values=naValues, )

        # if jsonData["rowsToDrop"] in list(dataSet.columns):
        #     dataSet = dataSet.drop(jsonData["rowsToDrop"], axis=1)

        dataSet = dataSet.drop(jsonData["rowsToDrop"], axis=1)

        self.predictionType = jsonData["predictionType"]
        self.rowsToDrop = jsonData["rowsToDrop"]
        self.dataClass = dataClass
        self.dataSet = dataSet
        self.continuousAttributes = continuousAttributes
        self.descreteAttributes = list(set(discreteAttributes) ^ set(jsonData["rowsToDrop"]))

        return self


