import MLAlgorithms.Utils.Numba.logistic_activation as la
from MLAlgorithms.NeuralNetwork.Node import Node
import numpy as np
import pandas as pd
from tqdm import tqdm
class NeuralNetwork:

    def __init__(self, train_data, number_of_hidden_layers, nodes_per_hidden_layer, prediction_type, unknown_col='class'):

        self.train_data = train_data
        self.predictionType = prediction_type

        self.activation_dict = {}
        self.activation_dict["logistic"] = (la.activate, la.activation_derivative)

        
        
        #This way the unknowns can be passed in as either part of the data frame or as a separate list
        #If the unknowns are part of the training set, they are split into their own Series
        list_types = (list, tuple, np.ndarray, pd.Series)
        if isinstance(unknown_col, list_types):
            self.unknown_col = pd.Series(unknown_col)
        elif isinstance(unknown_col, str):
           
            self.unknown_col = self.train_data[unknown_col][:]
            self.train_data = self.train_data.drop(unknown_col, axis=1)


        # self.unknown_col.reset_index(drop=True)
        # self.train_data.reset_index(drop=True)
        

        self._create_network(self.train_data, number_of_hidden_layers, nodes_per_hidden_layer, prediction_type)
        self.prev_update = np.zeros_like(self.weight_matrix)



    def _feed_forward(self, batch):
        """
        docstring
        """

        input_data = batch.to_numpy(dtype=np.longdouble)
        layer_input = input_data # layer_input initally contains a 2D array of every value for each attribute

        total_layers = len(self.layerDict.keys())

        self.derivative_matrix = np.ones((batch.shape[0], self.weight_matrix.shape[0], self.weight_matrix.shape[1]), dtype=np.longdouble)

        
        for layer_num in range(1,total_layers):
            prev_layer_indices = [node.index for node in self.layerDict[layer_num-1]]
            layer_indices = [node.index for node in self.layerDict[layer_num]]

            weights = self.weight_matrix[min(prev_layer_indices):max(prev_layer_indices)+1, min(layer_indices):max(layer_indices)+1]

            temp = (layer_input@weights)[:] # Used to apply the activation function in the next layer 
            
 

            for i, node in enumerate(self.layerDict[layer_num]):
                # print(layer_num, i)
                derivatives = node.activation_function_derivative(temp[:, i])
                self.derivative_matrix[:, min(prev_layer_indices):max(prev_layer_indices)+1, min(layer_indices):max(layer_indices)+1] = np.outer(weights, derivatives).reshape(weights.shape[0], -1, len(derivatives)).transpose(2,0,1)
                temp[:,i] = node.activation_function(temp[:,i])

            layer_input = temp  

        if self.predictionType == "classification":
            output = np.zeros_like(layer_input)
            for i, row in enumerate(output):
                exp_arr = np.exp(layer_input[i])
                output[i] = exp_arr/(exp_arr.sum())
            return output
        else:
            return layer_input
                
        
    def _backpropagate(self, learning_rate=0.01, batch_size=0.1):
        """
        Runs one iteration of backprop
        """
        batch = self.train_data.sample(n=batch_size)
        predicted = self._feed_forward(batch)
        # print(predicted.shape)
        cost_function = (predicted - self.unknown_col[batch.index])**2
        # return cost_function
        dCost_function = -2*(predicted-self.unknown_col[batch.index])
        if self.predictionType == "classification":
            dCost_function *= predicted
        

        update_matrix = np.ones_like(self.weight_matrix)

        total_layers = len(self.layerDict.keys())
        right_layer_cost = dCost_function
        
        for layer_num in reversed(range(1,total_layers)):
            # print("\nCurrent Layer Num: ",layer_num)
            left_layer_indices = [node.index for node in self.layerDict[layer_num-1]]
            layer_indices = [node.index for node in self.layerDict[layer_num]]
            # print("Layer Indices: ", layer_indices)
            # print("Left Indices: ", left_layer_indices)
            


            for i, node in enumerate(self.layerDict[layer_num]):
                # print(f"============= Node {node.cordinate} - Index {node.index} ===========")
                partial_derivative = self.derivative_matrix[:, min(left_layer_indices):max(left_layer_indices)+1, node.index]
                # print("RightLayer: ", right_layer_cost.shape)
                # print("Partial: ", partial_derivative.shape)
                # print("Weights: ", self.weight_matrix[left_layer_indices, node.index].shape)
                # print("Update: ", update_matrix[min(left_layer_indices):max(left_layer_indices)+1, node.index].shape)
                # print("Inner 1: ", np.inner(right_layer_cost.T, partial_derivative.T).shape)
                # print("Inner 2: ", np.inner(self.weight_matrix[left_layer_indices, node.index], np.inner(right_layer_cost.T, partial_derivative.T)).shape)
                update_matrix[min(left_layer_indices):max(left_layer_indices)+1, 
                node.index] = self.weight_matrix[min(left_layer_indices):max(left_layer_indices)+1, node.index] * np.inner(right_layer_cost[:,i].T, partial_derivative.T)
                # print("======================")
                

            #Update right_layer_cost
            #print(update_matrix[min(left_layer_indices):max(left_layer_indices)+1, min(layer_indices):max(layer_indices)+1])
            # print("Layer Update Shape: ", update_matrix[min(left_layer_indices):max(left_layer_indices)+1, min(layer_indices):max(layer_indices)+1].shape)
            right_layer_cost = np.matmul(right_layer_cost, update_matrix[min(left_layer_indices):max(left_layer_indices)+1, min(layer_indices):max(layer_indices)+1].T)
        
        # print("Previous Matrix")
        # print(self.weight_matrix)
        self.weight_matrix = ((1.0*learning_rate)*update_matrix + (0.5*learning_rate)*self.prev_update) + self.weight_matrix
        # print("Current Matrix")
        # print(self.weight_matrix)    
        
        self.prev_update = update_matrix
        






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
            self.layerDict[0].append(Node(node_index, (0, node), self.activation_dict["logistic"]))
            node_index += 1
        
        #Handles Hidden Layers
        for layer in range(number_of_hidden_layers):
            self.layerDict[layer+1] = []
            for node_num in range(nodes_per_hidden_layer[layer]):
                self.layerDict[layer+1].append(Node(node_index, (layer+1, node_num), self.activation_dict["logistic"]))
                node_index += 1
        
        #Handle Output
        curr_layer = number_of_hidden_layers + 1
        self.layerDict[curr_layer] = []
        if prediction_type == "classification":
            for unk in enumerate(self.unknown_col.iloc[0]):
                self.layerDict[curr_layer].append(Node(node_index, (curr_layer, unk[0]), self.activation_dict["logistic"]))
                node_index += 1
            temp_unk = np.zeros((len(self.unknown_col), len(self.unknown_col.iloc[0])), dtype=np.float64)
            for i, row in enumerate(temp_unk):
                temp_unk[i] = self.unknown_col.iloc[i]
            
            self.unknown_col = temp_unk
        else:
            self.layerDict[curr_layer].append(Node(node_index, (curr_layer, 0), self.activation_dict["logistic"]))
                                    

        #Initializing Weights:
        self.weight_matrix = np.triu(np.random.uniform(-0.1, 0.1, size=(node_index, node_index)), 1)
        self.derivative_matrix = np.ones((input_data.shape[0], self.weight_matrix.shape[0], self.weight_matrix.shape[1]))

