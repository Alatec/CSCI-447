import MLAlgorithms.Utils.Numba.logistic_activation as lga
import MLAlgorithms.Utils.Numba.linear_activation as lia
from MLAlgorithms.NeuralNetwork.Node import Node
from MLAlgorithms.Utils.OneHotEncoder import OneHotEncoder
from MLAlgorithms.GeneticAlgorithms.particle import Particle
from MLAlgorithms.GeneticAlgorithms.differential_mutation import differential_mutation
from MLAlgorithms.GeneticAlgorithms.differential_mutation import differential_binomial_crossover

import random as rand
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm


""" Neural Network Notes:
    Currently, this Network does not have any bias.

"""


class NeuralNetwork:

    def __init__(self, train_data, number_of_hidden_layers, nodes_per_hidden_layer, prediction_type, unknown_col='class', is_regression_data=False, seed=0):
        """
        self.train_data: Encoded Pandas DataFrame (unknown column included)
        self.predictionType: String representing the prediction type (regression || classification)
        self.unknown_col: String or Pandas Series containing truth values for the dataset (If string, must be column in train_data)
        
        self.activation_dict: A containing the activation functions and activation function derivatives for a given activation function type
        """

        np.random.seed(seed)
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
                self.unknown_col[self.unknown_col==2] = 0
                self.unknown_col[self.unknown_col==4] = 1
                self.unknown_df["unknown"] = self.unknown_col
        
            
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
                ret_array[output<thresh] = 2 #self.ohe.encodedDict["unknown"][0][0]
                ret_array[output>=thresh] = 4 #self.ohe.encodedDict["unknown"][0][1]
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
        # weight_shape = self.weight_matrix.shape
        # temp = self.weight_matrix.flatten()
        
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
        batch = self.train_data.sample(n=batch_size, random_state=(69+self.random_constant))
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
            left_layer_indices = [node.index for node in self.layerDict[layer_num-1]]
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
        
        self.weight_matrix = ((0.9*learning_rate)*update_matrix + (0.1*learning_rate)*self.prev_update) + self.weight_matrix
        

        
        self.prev_update = copy.deepcopy(update_matrix)
        
    def _particle_swarm_optimize(self, particle_count, max_iter=1000, batch_size=0.1, cost_func="321654"):
        print("Initializing Particles:")
        particles = [Particle(self.weight_matrix, index=i) for i in range(particle_count)]

        print("Calculating Fitness")
        best_fit = 1e6
        global_best = np.zeros_like(self.weight_matrix)
        fitness_matrix = np.zeros((max_iter,3))
        new_best_count = 0

        # Iterate through each individual
        for i in tqdm(range(max_iter)):
            average = 0
            batch = self.train_data.sample(frac=batch_size, random_state=(69+self.random_constant))
            self.random_constant += 1
            truths = self.unknown_df.loc[batch.index].to_numpy()

            # Update particle's position
            for part in particles:
                self.weight_matrix = part.position
                predicted = self._feed_forward(batch, testing=True)
                fitness = part.evalutate(predicted, truths, cost_func)
                average += fitness
            
                # GBest calculation
                if fitness < best_fit:
                    best_fit = fitness
                    global_best = part.position[:]
                    new_best_count += 1
                
            fitness_matrix[i,0] = np.abs(global_best.flatten()).max()
            fitness_matrix[i,1] = np.abs(global_best.flatten()).min()
            fitness_matrix[i,2] = average/particle_count
                
            # Apply velocity for each particle based on global best
            for part in particles:
                part.accelerate(global_best)
        
        # Return best
        self.weight_matrix = global_best
        print(f"New Best Count: {new_best_count}")
        return fitness_matrix



    def genetic_algorithm(self, population_size, maxItter, batch_size, mutation_rate, num_cuts = 10, cost_func='bin_cross'):

        # Create population of random weights
        population = [np.random.uniform(-0.1, 0.1, self.weight_matrix.shape) for i in range(population_size)]
        prev_result = 0

        #List of indices to select as cutpoints
        cutpoint_selection = np.arange(len(self.weight_matrix.flatten())-1)

        # Initialize best fit parameters
        best_fitness = 1e6
        best_fit_matrix = self.weight_matrix[:]

        fitnesses = np.zeros((maxItter,3))

        # Main for loop for iterating through the generations
        for i in tqdm(range(maxItter)):

            # Create 1 batch for every organism to test with
            batch = self.train_data.sample(frac=batch_size, random_state=(69+self.random_constant))
            self.random_constant += 1
            truths = self.unknown_col[batch.index].to_numpy()
            current_result = 0
            parent_fitness = np.zeros(len(population))

            next_generation = []

            # Evaluate fitness of Parents
            for j, parent in enumerate(population):
                
                
                self.weight_matrix = parent
                
                parent_fitness[j] = self._evolution_fitness(cost_func=cost_func).sum() #TODO

                if parent_fitness[j] < best_fitness:
                    best_fitness = parent_fitness[j]
                    best_fit_matrix = parent[:]

            fitnesses[i,0] = np.abs(best_fit_matrix.flatten()).max()
            fitnesses[i,1] = np.abs(best_fit_matrix.flatten()).min()
            fitnesses[i,2] = parent_fitness.mean()
            

            # SELECTION
            # Sort parents by fitness
            chads = np.argsort(parent_fitness)
            
            
            selection_prob = np.arange(len(chads), dtype=np.float)[::-1]
            
            selection_prob[:len(chads)//2]+=5
            
            selection_prob /= selection_prob.sum()
            chads = np.random.choice(chads, len(chads)//2, p=selection_prob, replace=False)
            parent_shape = population[0].shape

            # CROSSOVER
            # Generate children
            for j, chad in enumerate(chads[:-1]):
                parent1 = population[chad].flatten()
                parent2 = population[chads[j+1]].flatten()
                
                cutpoints = np.sort(np.random.choice(cutpoint_selection, num_cuts, replace=False).astype(np.int))

                child1 = np.zeros_like(parent1)
                child2 = np.zeros_like(parent1)

                # Each child gets genetic material from parents up to the first cut point
                child1[:cutpoints[0]] = parent2[:cutpoints[0]]
                child2[:cutpoints[0]] = parent1[:cutpoints[0]]


                # For each cut, give children chromosones. Alternate each itteration 
                for k, cut in enumerate(cutpoints[:-1]):
                    if k%2 == 0:
                        child1[cut:cutpoints[k+1]] = parent1[cut:cutpoints[k+1]]
                        child2[cut:cutpoints[k+1]] = parent2[cut:cutpoints[k+1]]
                    else:
                        child1[cut:cutpoints[k+1]] = parent2[cut:cutpoints[k+1]]
                        child2[cut:cutpoints[k+1]] = parent1[cut:cutpoints[k+1]]

                # Remaining genetic material is given to the children
                child1[cutpoints[-1]:] = parent1[cutpoints[-1]:]
                child2[cutpoints[-1]:] = parent2[cutpoints[-1]:]

                # Mutate the children. 
                child1_mutated_genes = np.random.choice([0,1], child1.shape, p=[1-mutation_rate,mutation_rate])
                child1_mutation_amount = np.random.uniform(0.9,1.1,child1.shape)
                child1[child1_mutated_genes==1] *= child1_mutation_amount[child1_mutated_genes==1]

                child2_mutated_genes = np.random.choice([0,1], child2.shape, p=[1-mutation_rate,mutation_rate])
                child2_mutation_amount = np.random.uniform(0.9,1.1,child2.shape)
                child2[child2_mutated_genes==1] *= child2_mutation_amount[child2_mutated_genes==1]

                # Add the children to the next generation
                next_generation.append(np.reshape(child1, parent_shape))
                next_generation.append(np.reshape(child2, parent_shape))
            
            #CROSSOVER
            #Edge Case for final set of parents
            parent1 = population[chads[-1]].flatten()
            parent2 = population[chads[0]].flatten()
            
            cutpoints = np.sort(np.random.choice(cutpoint_selection, num_cuts, replace=False))

            child1 = np.zeros_like(parent1)
            child2 = np.zeros_like(parent1)

            child1[:cutpoints[0]] = parent1[:cutpoints[0]]
            child2[:cutpoints[0]] = parent2[:cutpoints[0]]

            for k, cut in enumerate(cutpoints[:-1]):
                if k%2 == 0:
                    child1[cut:cutpoints[k+1]] = parent1[cut:cutpoints[k+1]]
                    child2[cut:cutpoints[k+1]] = parent2[cut:cutpoints[k+1]]
                else:
                    child1[cut:cutpoints[k+1]] = parent2[cut:cutpoints[k+1]]
                    child2[cut:cutpoints[k+1]] = parent1[cut:cutpoints[k+1]]
            
            child1[cutpoints[-1]:] = parent1[cutpoints[-1]:]
            child2[cutpoints[-1]:] = parent2[cutpoints[-1]:]

            # MUTATION
            child1_mutated_genes = np.random.choice([0,1], child1.shape, p=[1-mutation_rate,mutation_rate])
            child1_mutation_amount = np.random.uniform(0.9,1.1,child1.shape)
            child1[child1_mutated_genes==1] *= child1_mutation_amount[child1_mutated_genes==1]

            child2_mutated_genes = np.random.choice([0,1], child2.shape, p=[1-mutation_rate,mutation_rate])
            child2_mutation_amount = np.random.uniform(0.9,1.1,child2.shape)
            child2[child2_mutated_genes==1] *= child1_mutation_amount[child2_mutated_genes==1]

            next_generation.append(np.reshape(child1, parent_shape))
            next_generation.append(np.reshape(child2, parent_shape))

            population = next_generation[:]

        self.weight_matrix = best_fit_matrix
        return fitnesses
        

    def _evolution_fitness(self, batch_size=0.69, cost_func='multi_cross'):
        batch = self.train_data.sample(frac=batch_size, random_state=(69+self.random_constant))
        predicted = self._feed_forward(batch)
        truths = self.unknown_df.loc[batch.index].to_numpy()
        delta = 1
        p = 1.75
        # self.random_constant += 1

        # Binary Cross Entropy Loss
        if cost_func == 'bin_cross':
      
            output = np.zeros_like(predicted)
            output[truths==1] = -np.log(predicted[truths==1]+0.001)
            output[truths==0] = -np.log(1.001-predicted[truths==0])
            Cost_function = output
        
        elif cost_func == 'MAE':
            
             #*dPredicted w.r.t weights)
            Cost_function = np.abs(truths-predicted)
        elif cost_func == 'huber':
            Cost_function = np.where(np.abs(truths-predicted) < delta , 0.5*((truths-predicted)**2), delta*np.abs(truths - predicted) - 0.5*(delta**2))
        elif cost_func == 'log_cosh':
           

            Cost_function = np.log(np.cosh(predicted - truths))
        elif cost_func == 'tweedie':
            # predicted = np.abs(predicted)
            Cost_function = truths * np.sign(predicted) * np.power(np.abs(predicted), 1-p)/(1-p) + np.sign(predicted) * np.power(np.abs(predicted), 2-p)/(2-p)
        elif cost_func == 'arb_pdf':
            truth_pdf, _ = np.histogram(truths, density=True)

            Cost_function = -truths*1
        else:
            #Quadratic Loss 
             #*dPredicted w.r.t weights)
            Cost_function = (predicted-self.unknown_df.loc[batch.index].to_numpy())**2
             #*dPredicted w.r.t weights

        return Cost_function

    def differential_evolution(self, population_size, maxIter, batch_size, mutation_rate, cross_over_prob, cost_func = 'bin_cross'):
        # Create population of random weights
        population = np.array([np.random.uniform(-0.1, 0.1, self.weight_matrix.shape) for i in range(population_size)])
        i = 0
        prev_result = 0
        seed_counter = 0
        fitnesses = np.zeros((maxIter,3))
        for i in tqdm(range(maxIter)):
            current_result = 0

            for j in range(len(population)):
                self.random_constant += 1
                random_individual_index = rand.sample([x for x in range(population_size) if x != j], 3)
                # Check fitness
                self.weight_matrix = population[j]
                parent_fitness = self._evolution_fitness(cost_func=cost_func)
                # Create Trial
                trial_matrix = differential_mutation(population[j], 
                                                    population[random_individual_index[0]], 
                                                    population[random_individual_index[1]], 
                                                    population[random_individual_index[2]],
                                                    mutation_rate)

                # Create offspring
                child_matrix = differential_binomial_crossover(population[j], trial_matrix, cross_over_prob, seed=seed_counter)
                seed_counter += 1

                self.weight_matrix = child_matrix
                child_fitness = self._evolution_fitness(cost_func=cost_func)

                final_parent_fitness = parent_fitness.sum().sum()
                fitnesses[i,2] += final_parent_fitness

                final_child_fitness = child_fitness.sum().sum()
                if final_child_fitness < final_parent_fitness:
                    population[j] = child_matrix


                final_fitness = final_child_fitness if final_child_fitness < final_parent_fitness else final_parent_fitness
                current_result += final_fitness

            fitnesses[i,0] = 17
            fitnesses[i,1] = 17
            fitnesses[i,2] /= len(population)

            if ((abs(current_result - prev_result) / abs(current_result + prev_result)) > .0000001):
                prev_result = current_result
            else: 
                break

        
        best = population[0]
        self.weight_matrix = best
        best_fitness = self._evolution_fitness(cost_func=cost_func).sum()
        for individual in population:

            self.weight_matrix = individual
            individual_fitness = self._evolution_fitness(cost_func=cost_func).sum()

            if individual_fitness < best_fitness:
                best = individual[:]
                best_fitness = individual_fitness
        
        return fitnesses


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
        self.weight_matrix = np.random.uniform(-.1, .1, size=(node_index, node_index))
        self.derivative_matrix = np.ones((input_data.shape[0], self.weight_matrix.shape[0], self.weight_matrix.shape[1]))
        self.prev_update = np.zeros_like(self.weight_matrix)

