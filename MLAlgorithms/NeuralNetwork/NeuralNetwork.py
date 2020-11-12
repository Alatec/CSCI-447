import MLAlgorithms.Utils.Numba.logistic_activation as lga
import MLAlgorithms.Utils.Numba.linear_activation as lia
from MLAlgorithms.NeuralNetwork.Node import Node
from MLAlgorithms.Utils.OneHotEncoder import OneHotEncoder

import numpy as np
import pandas as pd
from tqdm import tqdm
# np.random.seed(seed=420)



""" Neural Network Notes:
    Currently, this Network does not have any bias.

"""


class NeuralNetwork:

    def __init__(self, train_data, number_of_hidden_layers, nodes_per_hidden_layer, prediction_type, unknown_col='class', is_regression_data=False):
        """
        self.train_data: Encoded Pandas DataFrame (unknown column included)
        self.predictionType: String representing the prediction type (regression || classification)
        self.unknown_col: String or Pandas Series containing truth values for the dataset (If string, must be column in train_data)
        
        self.activation_dict: A containing the activation functions and activation function derivatives for a given activation function type
        """

        # np.random.seed(69)
        self.train_data = train_data
        self.predictionType = prediction_type
        self.activation_dict = {}
        self.activation_dict["logistic"] = (lga.activate, lga.activation_derivative)
        self.activation_dict["linear"] = (lia.activate, lia.activation_derivative)
        self.activation_dict["bias"] = (lambda x: 1, lambda x: 1)
        self.random_constant = 0
        


        
        
        # This way, the unknowns can be passed in as either part of the data frame or as a separate list
        # If the unknowns are part of the training set, they are split into their own Series
        list_types = (list, tuple, np.ndarray, pd.Series)
        if isinstance(unknown_col, list_types):
            self.unknown_col = pd.Series(unknown_col)
        elif isinstance(unknown_col, str):
            self.unknown_col = self.train_data[unknown_col][:]
            self.train_data = self.train_data.drop(unknown_col, axis=1)
        
        self.unknown_df = pd.DataFrame()
        self.unknown_df['unknown'] = self.unknown_col
        self.unknown_df = self.unknown_df.reset_index(drop=True)
        if prediction_type == 'classification':
            self.ohe = OneHotEncoder()
            self.unknown_df = self.ohe.train_fit(self.unknown_df, ["unknown"])
            if self.unknown_df.shape[1] == 1:
                self.unknown_col = self.unknown_df["unknown"]
        
            
        # On object creation, create a graph to resemble the Neural Network
        self._create_network(self.train_data, number_of_hidden_layers, nodes_per_hidden_layer, prediction_type)
        
    def test(self, dataset, thresh="mean"):
        """
        dataset: Pandas DataFrame to predict with
        thresh: (str or float) threshold to determine positive class
        """
        output = self._feed_forward(dataset, testing=True)
        if self.predictionType == 'classification':
            
            # Binary Classification
            if output.shape[1] == 1:
                ret_array = np.zeros(output.shape[0])
                output = output.reshape(output.shape[0])
                if thresh == "median": thresh = np.median(output)
                elif thresh == "mean": thresh = np.mean(output)
                ret_array[output<thresh] = self.ohe.encodedDict["unknown"][0][0]
                ret_array[output>=thresh] = self.ohe.encodedDict["unknown"][0][1]
                return ret_array
            else:
                output_cols = output.argmax(axis=1)
                unknown_cols = self.unknown_df.columns
                ret_array = []
                for val in output_cols:
                    col = unknown_cols[val]

                    for x in self.ohe.encodedDict["unknown"]:
                        if len(x) > 1:
                            if x[1] == col:
                                ret_array.append(x[0])
                                break
                            continue
                        ret_array.append("unknown")
                return ret_array
        else:
            return output

    def train(self, maxIter, learning_rate, batch_size):
        """
        maxIter: Number of times to run backprop
        learning_rate: Learning rate of backprop
        batch_size: The size of batch to use as a percentage of training set size
        """
        # Run Backprop maxIter number of times
        for i in range(maxIter):
            self._back_propagate(learning_rate=learning_rate, batch_size=batch_size)

    def _feed_forward(self, batch, testing=False):
        """
        batch: Sampled Pandas DataFrame

        _feed_forward walks through the neural network

        Some of the logic for back propagation is placed in the feed forward step to make use of dynamic programming
        """

        input_data = batch.to_numpy(dtype=np.longdouble)
        layer_input = input_data # layer_input initally contains a 2D array of every value for each attribute
        total_layers = len(self.layerDict.keys())
        
        if not testing:
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

                if not testing:
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
                

        # If classification, apply softmax
        if self.predictionType == "classification" and self.unknown_df.shape[1] > 1:
            output = np.zeros_like(layer_input, dtype=np.float64)
            for i, row in enumerate(output):
                exp_arr = np.exp(layer_input[i]-layer_input[i].max())
                output[i] = exp_arr/(exp_arr.sum())
            return output
        else:
            return layer_input

                
        
    def _back_propagate(self, learning_rate=0.1, batch_size=0.69, cost_func='multi_cross'):
        """
        learning_rate: float - Used to describe the rate the Neural Network runs
        batch_size: int - Number of points grabbed from the data set

        _back_propagate is used to update the edge weights
        In order to update the edge weights, _back_propagate multiplies a chain of derivative_matrix look-ups. These derivative_matrices are made in _feed_forward  

        returns output of cost_function
        """
        batch = self.train_data.sample(frac=batch_size, random_state=(69+self.random_constant))
        self.random_constant += 1

        
        # These are all the different loss functions we use

        # Binary Cross Entropy Loss
        if cost_func == 'bin_cross':
            predicted = self._feed_forward(batch).T
            truths = self.unknown_col[batch.index].to_numpy()
            
            dCost_function = (1/len(batch))* (np.divide(truths,(predicted+0.0001)) + np.divide(1-truths,(1.0001-predicted))).T
        # Multi-Class Cross Entropy Loss
        elif cost_func == 'multi_cross':
            predicted = self._feed_forward(batch)
            truths = self.unknown_df.loc[batch.index].to_numpy().argmax(axis=1)
            #https://deepnotes.io/softmax-crossentropy
            m = truths.shape[0]
            predicted[range(m),truths] -= 1

            dCost_function = -(predicted/m)
            


        else:
            #Quadratic Loss 
            predicted = self._feed_forward(batch)
            dCost_function = -1*np.abs(predicted-self.unknown_df.loc[batch.index].to_numpy())
             #*dPredicted w.r.t weights
            
        update_matrix = np.zeros_like(self.weight_matrix)

        total_layers = len(self.layerDict.keys())
        right_layer_cost = dCost_function
        
        for layer_num in reversed(range(1,total_layers)):
            left_layer_indices = []
            left_layer_indices = [node.index for node in self.layerDict[layer_num-1]]
            layer_indices = []
            layer_indices = [node.index for node in self.layerDict[layer_num]]

            for i, node in enumerate(self.layerDict[layer_num]):

                partial_derivative = self.derivative_matrix[:, min(left_layer_indices):max(left_layer_indices)+1, node.index]

                # Matrix slicing notation changes if there is only 1 node in a layer
                if len(self.layerDict[layer_num]) == 1:
                    update_matrix[min(left_layer_indices):max(left_layer_indices)+1,  node.index] =  right_layer_cost.T @ partial_derivative
                else:
                    update_matrix[min(left_layer_indices):max(left_layer_indices)+1,  node.index] =  np.inner(right_layer_cost[:,i].T, partial_derivative.T)

            
            #Update right_layer_cost
            right_layer_cost = np.matmul(right_layer_cost, update_matrix[min(left_layer_indices):max(left_layer_indices)+1, min(layer_indices):max(layer_indices)+1].T)
        
        self.update_matrix = (update_matrix-update_matrix-update_matrix.min())/(update_matrix.max()-update_matrix.min()) # This line is for video purposes only
        update_matrix = (update_matrix-update_matrix-update_matrix.min())/(update_matrix.max()-update_matrix.min())
        self.weight_matrix = ((0.9*learning_rate)*update_matrix + (0.1*learning_rate)*self.prev_update) + self.weight_matrix
        

        
        self.prev_update = update_matrix[:]
        

    def differential_evolution_fitness(self, learning_rate=0.1, batch_size=0.69, cost_func='multi_cross'):
        batch = self.train_data.sample(frac=batch_size, random_state=(69+self.random_constant))
        self.random_constant += 1

        # Binary Cross Entropy Loss
        if cost_func == 'bin_cross':
            predicted = self._feed_forward(batch).T
            truths = self.unknown_col[batch.index].to_numpy()
            
            dCost_function = (1/len(batch))* (np.divide(truths,(predicted+0.0001)) + np.divide(1-truths,(1.0001-predicted))).T
        
        else:
            #Quadratic Loss 
            predicted = self._feed_forward(batch)
            dCost_function = -1*np.abs(predicted-self.unknown_df.loc[batch.index].to_numpy())
             #*dPredicted w.r.t weights

        return dCost_function
        
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
            self.layerDict[0].append(Node(node_index, (0, node), self.activation_dict["linear"]))
            node_index += 1
        
        #Handles Hidden Layers
        for layer in range(number_of_hidden_layers):
            self.layerDict[layer+1] = []
            for node_num in range(nodes_per_hidden_layer[layer]):
                self.layerDict[layer+1].append(Node(node_index, (layer+1, node_num), self.activation_dict["logistic"]))
                node_index += 1
            # This is the bias node - Not Used in project 3
            # self.layerDict[layer+1].append(Node(node_index, (layer+1, node_num), self.activation_dict["bias"]))
            # node_index += 1
        
        #Handle Output
        curr_layer = number_of_hidden_layers + 1
        self.layerDict[curr_layer] = []
        if prediction_type == "classification":
            # If the dataset is a binary classification
            if self.unknown_df.shape[1] == 1:
                self.layerDict[curr_layer].append(Node(node_index, (curr_layer, 0), self.activation_dict["logistic"]))
                node_index += 1
                
            else:
                #Apply Linear activation function to output layer of multi classification datasets
                for unk in enumerate(self.unknown_df.iloc[0]):
                    self.layerDict[curr_layer].append(Node(node_index, (curr_layer, unk[0]), self.activation_dict["linear"]))
                    node_index += 1

        else:
            #Apply Linear activation to the output of regression datasets
            self.layerDict[curr_layer].append(Node(node_index, (curr_layer, 0), self.activation_dict["linear"]))
            node_index += 1
                                    

        #Initializing Weights:
        self.weight_matrix = np.random.uniform(-0.1, 0.1, size=(node_index, node_index))
        self.derivative_matrix = np.ones((input_data.shape[0], self.weight_matrix.shape[0], self.weight_matrix.shape[1]))
        self.prev_update = np.zeros_like(self.weight_matrix)

