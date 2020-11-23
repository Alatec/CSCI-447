from MLAlgorithms.Utils.DistanceMatrix import DistanceMatrix
import numpy as np

"""
Clusters the data to be used for the various KNN algorithms with medoids

Args: 
    dataSet: Pandas DataFrame
    classfier: String
    discreteAttr: List<String>
    continAttr: List<String>
    predictionType: String
    k: int
    maxUter: int
    
Returns:
    dataSet.loc[medoidList]: Panadas DataFrame
"""
def KMediods(dataSet, classifier, discreteAttr, continAttr, predictionType, k, maxIter):
    # Pick k random cluster centers from the given data set
    mediods = _createMediods(dataSet, k)

    totalAttr = len(discreteAttr) + len(continAttr)
    percentDis = len(discreteAttr) / totalAttr
    percentCon = len(continAttr) / totalAttr

    iteration = 0

    # Create a matrix of scalar values between each data point. This is one up front heavy calculation that only
    # occurs once since each medoid is a data point.
    distanceMatrix = DistanceMatrix(dataSet, dataSet, continAttr, discreteAttr, percentCon, percentDis, predictionType, classifier)

    # Keep track of the medoid indices and non-medoids
    medoidList = np.array(mediods.index)
    nonMedoidList = np.array(list(set(dataSet.index)^set(medoidList)))

    # Create a binary matrix of each data point to keep track of the medoids
    mediodMatrix = np.zeros((len(dataSet), len(dataSet)), dtype=int)
    for medoid in medoidList:
        mediodMatrix[medoid][medoid] = medoid


    # Recalculate the medoids until there are no changes in medoids
    assignedClusters = {}
    while iteration < maxIter:
        print(iteration)
        oldMedoidList = np.copy(medoidList)

        # ========================================================================= Assign each point to a cluster

        if assignedClusters == {}:
            print("Assigning Clusters...")
            for nonMedoid in nonMedoidList:
                closestMedoid = medoidList[0]
                for medoid in medoidList:
                    if distanceMatrix.distanceMatrix[nonMedoid][medoid] < distanceMatrix.distanceMatrix[nonMedoid][closestMedoid]:
                        closestMedoid = medoid

                if closestMedoid not in assignedClusters:
                    assignedClusters[closestMedoid] = []
                assignedClusters[closestMedoid].append(nonMedoid)
            for medoid in medoidList:
                if medoid not in assignedClusters:
                    assignedClusters[medoid] = []



        # ====================================================================== calculate distortion
        print("Calculating Distortion...")
        initialDistortionSum = 0
        for mediodPoint in medoidList:
            for dataPoint in assignedClusters[mediodPoint]:
                initialDistortionSum += (distanceMatrix.distanceMatrix[dataPoint][mediodPoint])**2

        print("Distortion Calculated")
        # ============================================================= Recalculate our medoids

        print("Swapping...")
        # For each medoid and point assigned to each medoid, swap these points and calculate the new distortion
        for medoid, medoidRow in enumerate(medoidList):
            for dataPoint, dataRow in enumerate(dataSet.index):
                if medoid == dataPoint:
                    continue
                else:
                    tempMedoid = dataRow
                    tempDataPoint = medoidRow

                    # Before swapping, calculate the distortion for a temp swap.
                    newDistortionSum = 0
                    for mediodPointP in medoidList:

                        if mediodPointP == medoid:
                            mediodPointP = tempMedoid


                        for dataPointP in assignedClusters[medoidRow]:
                            if dataPointP == dataPoint:
                                dataPointP = tempDataPoint

                            newDistortionSum += (distanceMatrix.distanceMatrix[dataPointP][mediodPointP]) ** 2

                    # If the new distortion is greater than the initial distortion, actually swap points
                    if newDistortionSum > initialDistortionSum and tempMedoid not in medoidList:

                        assignedClusters[tempMedoid] = assignedClusters[medoidRow]
                        del assignedClusters[medoidRow]

                        medoidList[medoid] = tempMedoid
                        medoidRow = tempMedoid

        print("Swapping has Finished...")
        # =========================================================== Check if the medoids have changed
        iteration += 1
        # If the medoids have not changed, break. Otherwise, keep iterating
        if np.array_equal(oldMedoidList, medoidList):
            break

    return dataSet.loc[medoidList]


# This function creates k medoids
def _createMediods(dataSet, k):
    seed = 69
    mediods = dataSet.sample(k, random_state=seed)

    return mediods
