import numpy as np

def activate(input_value):
    input_value = np.clip(input_value, -1.0, 1.0)
    return 1/(1+np.exp(-input_value))

    
def activation_derivative(input_value):
    input_value = np.clip(input_value, -1.0, 1.0)
    return np.exp(input_value)/(1+np.exp(input_value))

        