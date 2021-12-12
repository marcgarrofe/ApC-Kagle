##################################################
## demo.py
## Demo document for "Aprenentatge Computacional" - Kaggle project
##################################################
__author__ = "Marc Garrofé Urrutia"
__license__ = "MIT"
__version__ = "1.0.1"
__date__ = "11/10/2021"
##################################################

# Imports :
import sys
import numpy as np

# Desactivem Warnings
import warnings
warnings.simplefilter("ignore")

import pickle
from sklearn.ensemble import RandomForestClassifier

# Constants
MODEL_BASE_PATH = '../models/'

def loadModel(modelPath):
    """
    Given the model name, returns de model
    :param modelPath: String with de models path
    :return: Model object
    """
    return pickle.load(open(modelPath, 'rb'))


def main(argv):
    """
    Given the terminal param array. Prints the classfication of the input data
    :param argv: Input Array Values
    :return: -1 if error
    """

    try:
        modelPath = str(MODEL_BASE_PATH) + str(argv[1])
        model = loadModel(modelPath)
    except:
        print("::-> Error in loadModel()")
        print("     FilePath = " + str(modelPath))
        return -1

    data = np.fromstring(argv[-1], sep=',').reshape(1, -1)
    result = model.predict(data)

    print("::-> Classfication = " + str(result))
    print("     Data = " + str(data))

main(sys.argv)