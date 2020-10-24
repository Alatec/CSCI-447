import numpy as np

def activate(input_value):
    return 1/(1+np.exp(-input_value))

    
def activation_derivative(input_value):
    print(input_value.dtype)
    return np.exp(input_value)/(1+np.exp(input_value))

        