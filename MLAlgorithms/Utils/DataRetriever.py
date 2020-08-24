import json, os, pandas as pd


class DataRetriever():
    def __init__(self, path):
        self.currentData = None
        self.data = None
        self.dataSetPath = os.path.split(path)[0]
        self.metaDataPath = path
        self.menu = []

        self._buildDataMenu()

    # Builds a menu from the meta data of a data set
    def _buildDataMenu(self):

        with open(self.metaDataPath) as f:
            self.data = json.load(f)

        for object in self.data:
            self.menu.append(object)

    def getDataMenu(self):
        return self.menu

    # Determines if given data exists in the menu
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

        dataSet = pd.read_csv(self.dataSetPath + "/" + dataPath, names=header, na_values=naValues)

        return dataSet

