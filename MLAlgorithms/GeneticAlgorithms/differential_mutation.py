import numpy as np
import random as rand

def differential_mutation(target_vector, x1, x2, x3, beta):
    if beta < 0 or beta > 2:
        print("Beta is not in a valid range")
        exit(-1)

    trial_vector = x1 + beta * (x2 - x3)

    return trial_vector


def differential_crossover(target_vector, trial_vector, p_sub_r):
    if len(target_vector) != len(trial_vector):
        print("The length of the target vector and trial vector must be equal")
        exit(-1)

    result_vector = np.zeros(len(target_vector))

    for i in range(len(target_vector)):
        rand_int = rand.randint(0, 1)
        if rand_int <= p_sub_r:
            result_vector[i] = target_vector[i]
        else:
            result_vector[i] = trial_vector[i]

    return result_vector



if __name__ == "__main__":
    rand.seed(69)

    p_sub_r = .5
    beta = 1

    target_vector = np.array([4, 2, 6, 7])
    x1 = np.array([8, 2, 4 ,1])
    x2 = np.array([1, 1, 8, 2])
    x3 = np.array([6, 2, 8, 1])

    trial_vector = differential_mutation(target_vector, x1, x2, x3, beta)

    print("The trial vector is ", trial_vector)

    result_vector = differential_crossover(target_vector, trial_vector, p_sub_r)

    print("The result vector is ", result_vector)