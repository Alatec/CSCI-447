import numpy as np





class Node:

    def __init__(self, index, cordinate, activation_function):
        """
        index: int
        cordinate: Tuple<int,int> layer, node
        activation_function: StringBOI
        """
        self.index = index
        self.cordinate = cordinate
        self.activation_function = activation_function[0]
        self.activation_function_derivative = activation_function[1]
        self.input_data = None

    def __str__(self):
        return f"Index: {self.index}\nCordinate: {self.cordinate}\nActivation Function: {self.activation_function}"

    

    
    """
    attr List<List<double>>: input_matrix
        Each element is a datapoint
            (Saving Activation Step) Each data point needs an array of derivatives of the input weights

    """
