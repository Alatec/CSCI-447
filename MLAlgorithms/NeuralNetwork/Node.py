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
        self.activation_function = activation_function

    def __str__(self):
        return f"Index: {self.index}\nCordinate: {self.cordinate}\nActivation Function: {self.activation_function}"

    
# TODO
    def activate(self, input_value):
        if self.activation_function == "logistic":
            return 1/(1+np.exp(-input_value))
        else:
            return input_value
        
