import numpy as np
import random as rand


def differential_mutation(target_vector, x1, x2, x3, beta):
    if beta < 0 or beta > 2:
        print("Beta is not in a valid range")
        exit(-1)

    trial_vector = x1 + beta * (x2 - x3)

    return trial_vector


def differential_binomial_crossover(target_vector, trial_vector, cross_over_prob, seed=0):
    rand.seed(seed)
    if len(target_vector) != len(trial_vector):
        print("The length of the target vector and trial vector must be equal")
        exit(-1)

    flat_target = target_vector.flatten()
    flat_trial = trial_vector.flatten()

    muts = np.random.choice([0, 1], flat_target.shape, p=[1-cross_over_prob, cross_over_prob])
    flat_target[muts == 1] = flat_trial[muts == 1]
    


    result_vector = np.reshape(flat_target, target_vector.shape)


    # result_vector = np.zeros(target_vector.shape)
    # for i in range(len(target_vector)):
    #     for j in range(len(target_vector)):

    #         rand_int = rand.randint(0, 1)
    #         if rand_int <= cross_over_prob:
    #             result_vector[i][j] = target_vector[i][j]
    #         else:
    #             result_vector[i][j] = trial_vector[i][j]


    return result_vector



if __name__ == "__main__":


    cross_over_prob = .5
    beta = 1

    target_vector = np.array([4, 2, 6, 7])
    x1 = np.array([8, 2, 4, 1])
    x2 = np.array([1, 1, 8, 2])
    x3 = np.array([6, 2, 8, 1])

    trial_vector = differential_mutation(target_vector, x1, x2, x3, beta)

    print("The trial vector is ", trial_vector)

    matrix1 = np.array([[0, 2, 4], [6, 8, 10]])
    matrix2 = np.array([[1, 3, 5], [7, 9, 11]])
    

    result_vector = differential_binomial_crossover(
        matrix1, matrix2, cross_over_prob)

    print("The result vector is ", result_vector)
