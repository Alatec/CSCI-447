from MLAlgorithms.NaiveBayes.naiveBayes import NaiveBayes
from MLAlgorithms.Utils.BinDiscretizer import BinDiscretizer
from MLAlgorithms.Utils.DataRetriever import DataRetriever
from MLAlgorithms.Utils.KFolds import KFolds
from MLAlgorithms.Utils.RangeNormalizer import RangeNormalizer
from MLAlgorithms.Utils.StandardNormalizer import StandardNormalizer


def calculateResults(predictions, answers):
    """Calculates the accuracy of the NaiveBayes algorithm

    Args:
        predictions (List<Predictions>): The result of the algorithm
        answers (List<Truths>): The answer to each row fed into the algorithm
    """
    t = 0
    f = 0
    for i in range(len(answers)):
        if predictions[i] == answers[i]:
            t += 1
            # print("SUCCESS: Prediction is {p1} and Answer is {a}".format(p1=predictions[i], a=answers[i]))
        else:
            f += 1
            # print("FAILURE: Prediction is {p1} and Answer is {a}".format(p1=predictions[i], a=answers[i]))

    print("The Percent True is {t}".format(t=t / len(answers)))
    print("The Percent False is {f}\n".format(f=f / len(answers)))

# For train. test in KFOLD:
#   binData()
#   NormalizeData
#   Naivebayes

dataRetriever = DataRetriever("../Datasets/metadata.json")

for dataSet in dataRetriever.getDataMenu():
    dataRetriever.retrieveData(dataSet)
    dataClass = dataRetriever.getDataClass()
    retrievedData = dataRetriever.getDataSet()

    print(f"CURRENTLY PRINTING RESULTS FOR THE DATASET {dataSet}")
    for train, test in KFolds(retrievedData, 10):
        trainBin = BinDiscretizer(train[dataRetriever.getContinuousAttributes()], multi=True)

        trainBin.train_multi()
        train[dataRetriever.getContinuousAttributes()] = trainBin.fit_multi(train[dataRetriever.getContinuousAttributes()] )
        test[dataRetriever.getContinuousAttributes()] = trainBin.fit_multi(test[dataRetriever.getContinuousAttributes()])

        # trainNormalized = RangeNormalizer(trainBin)
        # testNormalized = RangeNormalizer(testBin)

        # trainNormalized = trainNormalized.train_fit()
        # testNormalized = testNormalized.train_fit()

        naiveBayes = NaiveBayes(train, dataClass)

        answers = test[dataClass].to_numpy()[:]
        test = test.drop(columns=dataClass)
        predictions = naiveBayes.test(test)

        calculateResults(predictions, answers)




