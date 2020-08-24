import json, os, pandas as pd

class DataRetriever():
    def __init__(self, path):
        self.currentData = None
        self.data = None
        self.dataSetPath = os.path.split(path)[0]
        self.metaDataPath = path
        self.menu = []
        self.mostRecentJSON = None

        self._buildDataMenu()

    def _buildDataMenu(self):

        with open(self.metaDataPath) as f:
            self.data = json.load(f)

        for object in self.data:
            self.menu.append(object)

    def getDataMenu(self):
        return self.menu

    def hasData(self, data):
        return data in self.menu


    def retrieveData(self, data):
        if not self.hasData(data):
            return None
        with open(self.metaDataPath) as f:
            js = json.load(f)
            self.mostRecentJSON = js[data]

        dataPath = self.dataSetPath + "/" + self.mostRecentJSON["dataPath"]
        header = self.mostRecentJSON["attributes"]
        naValues = self.mostRecentJSON["NAValues"]

        dataSet = pd.read_csv(self.dataSetPath + "/" + dataPath, names=header, na_values=naValues)
        print(dataSet.head())


        return dataSet

