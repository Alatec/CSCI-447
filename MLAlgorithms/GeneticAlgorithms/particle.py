import numpy as np


class Particle:

    def __init__(self, initial_vector, vel_bound=.05, pos_bound=1.0, momentum=(2.05,2.05), index=0, vel_max = 0.05):
        self.position = np.random.uniform(-pos_bound, pos_bound, initial_vector.shape)
        self.velocity = np.random.uniform(-vel_bound, vel_bound, initial_vector.shape)
        self.pbest_vector = self.position[:]
        self.pbest_fitness = 1e6
        self.momentum = momentum
        self.phi = momentum[0]+momentum[1]
        self.X = 2/(self.phi-2+np.sqrt(self.phi*self.phi-4*self.phi))
        self.index=index
        self.vel_max = vel_max
    
    def accelerate(self, best):
        self.velocity = self.X*(self.velocity +np.random.uniform(0, self.momentum[0], self.velocity.shape)*(self.pbest_vector-self.position) +np.random.uniform(0, self.momentum[1], self.velocity.shape)*(best-self.position)) 
        self.velocity = np.clip(self.velocity, -self.vel_max, self.vel_max)

        self.position += self.velocity
        pass


    def evalutate(self, predicted, truths, cost_func='multi_cross'):
        fitness = 1e6
        delta = 1
        p = 1.75
         # Binary Cross Entropy Loss
        if cost_func == 'bin_cross':
            output = np.zeros_like(predicted)
            output[truths==1] = -np.log(predicted[truths==1]+0.001)
            output[truths==0] = -np.log(1.001-predicted[truths==0])
            fitness = np.mean(output)
        # Multi-Class Cross Entropy Loss
        elif cost_func == 'multi_cross':
            # https://deepnotes.io/softmax-crossentropy
            # TODO checkout out onehot
            log_likelihood = -np.log(predicted[:,truths])
            loss = np.sum(log_likelihood) / predicted.shape[0]
            return loss
        
        elif cost_func == "MAE":
            return np.abs(truths-predicted).sum()
        
        elif cost_func == 'huber':
            return np.where(np.abs(truths-predicted) < delta , 0.5*((truths-predicted)**2), delta*np.abs(truths - predicted) - 0.5*(delta**2)).sum().sum()
        elif cost_func == 'log_cosh':
           
            return np.log(np.cosh(predicted - truths)).sum()

        elif cost_func == 'tweedie':
            # predicted = np.abs(predicted)
            return (truths * np.sign(predicted) * np.power(np.abs(predicted), 1-p)/(1-p) + np.sign(predicted) * np.power(np.abs(predicted), 2-p)/(2-p)).sum().sum()
            
         

        else:
            #Quadratic Loss 
            output = ((predicted-truths)**2).flatten()
            fitness = np.mean(output)
            #*dPredicted w.r.t weights
            
            # if self.predictionType == "classification":
            #     dCost_function *= predicted

        if fitness < self.pbest_fitness:
            self.pbest_vector = self.position[:]
            self.pbest_fitness = fitness
        return fitness


    def __str__(self):
        return f"Index: {self.index} Best: {self.pbest_fitness}"

    
    """
    attr List<List<double>>: input_matrix
        Each element is a datapoint
            (Saving Activation Step) Each data point needs an array of derivatives of the input weights

    """
