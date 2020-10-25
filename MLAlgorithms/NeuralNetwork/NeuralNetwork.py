import MLAlgorithms.Utils.Numba.logistic_activation as la
from MLAlgorithms.NeuralNetwork.Node import Node
import numpy as np
import pandas as pd
from tqdm import tqdm

""" Questions

We are experiencing an overflow error with our sigmoid functions. How do we avoid this?
Currently, our Neural Network is learning to predict the most common classifcation. Why?

"""

""" Neural Network Notes:
    Currently, this Network does not have any bias.

"""


class NeuralNetwork:

    def __init__(self, train_data, number_of_hidden_layers, nodes_per_hidden_layer, prediction_type, unknown_col='class'):
        """
        self.train_data: Encoded Pandas DataFrame (unknown column included)
        self.predictionType: String representing the prediction type (regression || classification)
        self.unknown_col: String or Pandas Series containing truth values for the dataset (If string, must be column in train_data)
        
        self.activation_dict: A containing the activation functions and activation function derivatives for a given activation function type
        """


        self.train_data = train_data
        self.predictionType = prediction_type
        self.activation_dict = {}
        self.activation_dict["logistic"] = (la.activate, la.activation_derivative)

        
        
        # This way, the unknowns can be passed in as either part of the data frame or as a separate list
        # If the unknowns are part of the training set, they are split into their own Series
        list_types = (list, tuple, np.ndarray, pd.Series)
        if isinstance(unknown_col, list_types):
            self.unknown_col = pd.Series(unknown_col)
        elif isinstance(unknown_col, str):
            self.unknown_col = self.train_data[unknown_col][:]
            self.train_data = self.train_data.drop(unknown_col, axis=1)
        
        # On object creation, create a graph to resemble the Neural Network
        self._create_network(self.train_data, number_of_hidden_layers, nodes_per_hidden_layer, prediction_type)
        


    def _feed_forward(self, batch):
        """
        batch: Sampled Pandas DataFrame

        _feed_forward walks through the neural network

        Some of the logic for back propagation is placed in the feed forward step to make use of dynamic programming
        """

        input_data = batch.to_numpy(dtype=np.longdouble)
        layer_input = input_data # layer_input initally contains a 2D array of every value for each attribute
        total_layers = len(self.layerDict.keys())
        
        # Create a 3D matrix containing the partial derivative information of the weight matrix for each data point
        self.derivative_matrix = np.ones((batch.shape[0], self.weight_matrix.shape[0], self.weight_matrix.shape[1]), dtype=np.longdouble)

        # Iterate through each layer 
        for layer_num in range(1,total_layers):
            # Find each of indices of the previous layer and current layer in order to make our calculations
            left_layer_indices = [node.index for node in self.layerDict[layer_num-1]]
            layer_indices = [node.index for node in self.layerDict[layer_num]]

            # Select the portion of the weight matrix representing the edges between left_layer and the current layer
            weights = self.weight_matrix[min(left_layer_indices):max(left_layer_indices)+1, min(layer_indices):max(layer_indices)+1]

            # Used to apply the activation function in the next layer 
            layer_input = layer_input@weights
            
            # Iterate through each node in current layer
            for i, node in enumerate(self.layerDict[layer_num]):
                # Apply the activation function to the input data
                layer_input[:,i] = node.activation_function(layer_input[:,i])

                # This block of code is used to calculate the derivates for the upcoming back propagation step
                # A(Input_layer*Weight)
                # dA(Input_Layer*Weight)*Weight
                # =================================================================================================================================
                # Apply the derivative of the activation function to the input data
                derivatives = node.activation_function_derivative(layer_input[:, i])

                # Save paritial derivatives in the derivative matrix
                derivatives = np.outer(weights[:,i], derivatives).T
        
                # Update the selected portion of the derivative_matrix 
                self.derivative_matrix[:, min(left_layer_indices):max(left_layer_indices)+1, node.index] = derivatives
                # ==================================================================================================================================
                

        #If classification, apply softmax
        if self.predictionType == "classification":
            output = np.zeros_like(layer_input)
            for i, row in enumerate(output):
                exp_arr = np.exp(layer_input[i])
                output[i] = exp_arr/(exp_arr.sum())
            return output
        else:
            return layer_input
                
        
    def _back_propagate(self, learning_rate=0.01, batch_size=10):
        """
        learning_rate: float - Used to describe the rate the Neural Network runs
        batch_size: int - Number of points grabbed from the data set

        _back_propagate is used to update the edge weights
        In order to update the edge weights, _back_propagate multiplies a chain of derivative_matrix look-ups. These derivative_matrices are made in _feed_forward  

        returns output of cost_function
        """
        batch = self.train_data.sample(n=batch_size)
        predicted = self._feed_forward(batch)
        
        #Quadratic Loss 
        cost_function = (predicted - self.unknown_col[batch.index])**2
        
        
        dCost_function = -2*(predicted-self.unknown_col[batch.index]) #*dPredicted w.r.t weights
        # if self.predictionType == "classification":
        #     dCost_function *= predicted
        

        update_matrix = np.zeros_like(self.weight_matrix)

        total_layers = len(self.layerDict.keys())
        right_layer_cost = dCost_function
        
        for layer_num in reversed(range(1,total_layers)):

            left_layer_indices = [node.index for node in self.layerDict[layer_num-1]]
            layer_indices = [node.index for node in self.layerDict[layer_num]]

            


            for i, node in enumerate(self.layerDict[layer_num]):

                partial_derivative = self.derivative_matrix[:, min(left_layer_indices):max(left_layer_indices)+1, node.index]

                update_matrix[min(left_layer_indices):max(left_layer_indices)+1, 
                node.index] =  np.inner(right_layer_cost[:,i].T, partial_derivative.T)

            
            #Update right_layer_cost
            right_layer_cost = np.matmul(right_layer_cost, update_matrix[min(left_layer_indices):max(left_layer_indices)+1, min(layer_indices):max(layer_indices)+1].T)
        

        self.weight_matrix = ((1.0*learning_rate)*update_matrix + (0.1*learning_rate)*self.prev_update) + self.weight_matrix

        
        self.prev_update = update_matrix[:]
        
        return cost_function
        






    def _create_network(self, input_data, number_of_hidden_layers, nodes_per_hidden_layer, prediction_type):
        """
        input_data: Pandas DataFrame
        number_of_hidden_layers: int
        nodes_per_hidden_layer: List<int>
        prediction_type: String


        _create_network initializes the weight matrix and the adjacency dictionary for the Neural Network
        On network creation, each node gets assigned an activation function (As of right now, every node gets assigned the logistic activation function)

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
        self.weight_matrix = np.random.uniform(-0.1, 0.1, size=(node_index, node_index))
        self.derivative_matrix = np.ones((input_data.shape[0], self.weight_matrix.shape[0], self.weight_matrix.shape[1]))
        self.prev_update = np.zeros_like(self.weight_matrix)

