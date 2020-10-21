from MLAlgorithms.NeuralNetwork.Node import Node
import numpy as np
import pandas as pd
class NeuralNetwork:

    def __init__(self, train_data, number_of_hidden_layers, nodes_per_hidden_layer, prediction_type, unknown_col='class'):

        self.train_data = train_data

        self.predictionType = prediction_type


        
        #This way the unknowns can be passed in as either part of the data frame or as a separate list
        #If the unknowns are part of the training set, they are split into their own Series
        list_types = (list, tuple, np.ndarray, pd.Series)
        if isinstance(unknown_col, list_types):
            self.unknown_col = pd.Series(unknown_col)
        elif isinstance(unknown_col, str):
           
            self.unknown_col = self.train_data[unknown_col][:]
            self.train_data = self.train_data.drop(unknown_col, axis=1)


        self.unknown_col.reset_index(drop=True)
        self.train_data.reset_index(drop=True)
        

        self._create_network(self.train_data, number_of_hidden_layers, nodes_per_hidden_layer, prediction_type)



    def _feed_forward(self):
        """
        docstring
        """

        input_data = self.train_data.to_numpy()
        layer_input = input_data

        total_layers = len(self.layerDict.keys())

        for layer_num in range(1,total_layers):
            prev_layer_indices = [node.index for node in self.layerDict[layer_num-1]]
            layer_indices = [node.index for node in self.layerDict[layer_num]]

            weights = self.weight_matrix[min(prev_layer_indices):max(prev_layer_indices)+1, min(layer_indices):max(layer_indices)+1]

            layer_input = layer_input@weights
            print(layer_input.shape[1])
 

            for i, node in enumerate(self.layerDict[layer_num]):
                layer_input[:,i] = node.activate(layer_input[:,i])


        return layer_input
                
            
        # for each layer:

        #     temp = temp * weights[slicing_logic]
        #     for each node:
        #         temp[node] = node.activate(temp[node])
    


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
            self.layerDict[0].append(Node(node_index, (0, node), "logistic"))
            node_index += 1
        
        #Handles Hidden Layers
        for layer in range(number_of_hidden_layers):
            self.layerDict[layer+1] = []
            for node_num in range(nodes_per_hidden_layer[layer]):
                self.layerDict[layer+1].append(Node(node_index, (layer+1, node_num), "logistic"))
                node_index += 1
        
        #Handle Output
        curr_layer = number_of_hidden_layers + 1
        self.layerDict[curr_layer] = []
        if prediction_type == "classification":
            for unk in enumerate(self.unknown_col.unique()):
                self.layerDict[curr_layer].append(Node(node_index, (curr_layer, unk[0]), "logistic"))
                node_index += 1
        else:
            self.layerDict[curr_layer].append(Node(node_index, (curr_layer, 0), "logistic"))
                                    

        #Initializing Weights:
        self.weight_matrix = np.triu(np.random.uniform(-0.1, 0.1, size=(node_index, node_index)), 1)
