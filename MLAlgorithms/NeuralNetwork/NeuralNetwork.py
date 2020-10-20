from MLAlgorithms.NeuralNetwork.Node import Node
import numpy as np
import pandas as pd
class NeuralNetwork:

    def __init__(self, train_data, number_of_hidden_layers, nodes_per_hidden_layer, prediction_type, unknown_col='class'):
        # self.test_data = test
        self.train_data = train_data
        # self.k = k
        self.predictionType = prediction_type
        # self.contAttr = contAttr
        # self.discAttr = discAttr

        
        #This way the unknowns can be passed in as either part of the data frame or as a separate list
        #If the unknowns are part of the training set, they are split into their own Series
        list_types = (list, tuple, np.ndarray, pd.Series)
        if isinstance(unknown_col, list_types):
            self.unknown_col = pd.Series(unknown_col)
        elif isinstance(unknown_col, str):
           
            self.unknown_col = self.train_data[unknown_col][:]
            self.train_data = self.train_data.drop(unknown_col, axis=1)

        #Training the VDM requires the unknown column to be part of the set, so it is added to a temporary data frame
        self.unknown_col.reset_index(drop=True)
        self.train_data.reset_index(drop=True)
        # temp_train_with_unknown = self.train_data.copy(deep=True)
        # temp_train_with_unknown["unknown_col"] = self.unknown_col.values

        #Distance matrix is a tool for creating a weighted sum discrete and continous distances
        # self.distance_matrix = DistanceMatrix(self.test_data, temp_train_with_unknown, contAttr, discAttr, len(contAttr), len(discAttr), predictionType, "unknown_col")
        
        #self.neigbors is a len(test)xlen(train) array containing the distance from each point in test to each point in train
        # self.neighbors = self.distance_matrix.distanceMatrix 
        self._create_network( train_data, number_of_hidden_layers, nodes_per_hidden_layer, prediction_type)



    
    def _create_network(self, input_data, number_of_hidden_layers, nodes_per_hidden_layer, prediction_type):
        """
        input_data: Pandas DataFrame
        number_of_hidden_layers: int
        nodes_per_hidden_layer: List<int>
        prediction_type: String
        """

        self.layerDict = {}

        node_index = 0
        self.nodes_list = []
        #Handle Input Layer
        self.layerDict[0] = []

        for node in range(input_data.shape[1]):
            self.layerDict[0].append(Node(node_index, (0, node), "ree"))
            node_index += 1
        
        #Handles Hidden Layers
        for layer in range(number_of_hidden_layers):
            self.layerDict[layer+1] = []
            for node_num in range(nodes_per_hidden_layer[layer]):
                self.layerDict[layer+1].append(Node(node_index, (layer+1, node_num), "ree"))
                node_index += 1
        
        #Handle Output
        curr_layer = number_of_hidden_layers + 1
        self.layerDict[curr_layer] = []
        if prediction_type == "classification":
            for unk in enumerate(self.unknown_col.unique()):
                self.layerDict[curr_layer].append(Node(node_index, (curr_layer, unk[0]), "ree"))
                node_index += 1
        else:
            self.layerDict[curr_layer].append(Node(node_index, (curr_layer, 0), "ree"))
                                    

        #Initializing Weights:
        self.weight_matrix = np.random.uniform(-0.1, 0.1, size=(node_index, node_index))
