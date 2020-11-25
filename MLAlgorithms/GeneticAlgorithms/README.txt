Here is the structure for Project 4 - Genetic Algoirthms

    MLAlgorithms.GeneAlgorithms.differential_mutation.py
        Contains mutation and crossover operators used by DE

    MLAlgorithms.GeneAlgorithms.particle.py
        Contains information about an individual for PSO

    MLAlgorithms.GeneAlgorithms.de_testing_driver.py
        Driver used to test DE individually instead of using TheDriverToEndAllDriver.py
        #HIGHLY RECOMMEND THAT TheDriverToEndAllDriver.py IS TESTED FIRST#

    MLAlgorithms.GeneAlgorithms.ga_testing_driver.py
        Driver used to test GA individually instead of using TheDriverToEndAllDriver.py
        #HIGHLY RECOMMEND THAT TheDriverToEndAllDriver.py IS TESTED FIRST#

    MLAlgorithms.GeneAlgorithms.PSO_testing_driver.py
        Driver used to test PSO individually instead of using TheDriverToEndAllDriver.py
        #HIGHLY RECOMMEND THAT TheDriverToEndAllDriver.py IS TESTED FIRST#

    MLAlgorithms.GeneAlgorithms.Drivers.DE_driver.py
        Driver used to run data sets on DE
        This is consumed by TheDriverToEndAllDriver.py

    MLAlgorithms.GeneAlgorithms.Drivers.GA_driver.py
        Driver used to run data sets on GA
        This is consumed by TheDriverToEndAllDriver.py

    MLAlgorithms.GeneAlgorithms.Drivers.PSO_driver.py
        Driver used to run data sets on PSO
        This is consumed by TheDriverToEndAllDriver.py
    
    MLAlgorithms.GeneAlgorithms.Drivers.TheDriverToEndAllDriver.py
        This is the main driver we use to run every driver on each data set with three network architectures
        The settings used for this driver can be found in driver_params.json
        This file also dumps data out to DataDump directory

    MLAlgorithms.GeneAlgorithms.Drivers.DataAnaylizer.py
        Uses the data generated from TheDriverToEndAllDriver.py to create graphs

    MLAlgorithms.GeneAlgorithms.Drivers.driver_params.json
        The json file that contains all the settings for TheDriverToEndAllDriver.py



    MLAlgorithms.NeuralNetwork.NeuralNetwork.py
        Contains the neural network class
        Also contains each of the genetic algorithm implementations

    MLAlgorithms.NeuralNetwork.Node.py
        Contains the node class used by neural network

    MLAlgorithms.Utils.Numba.linear_activation.py
        Contains the linear_activation function used for a given node

    MLAlgorithms.Utils.Numba.logistic_activation.py
        Contains the logistic_activation function used for a given node
        
    MLAlgorithms.Utils.OneHotEncoder.py
        Used to encode categorical data