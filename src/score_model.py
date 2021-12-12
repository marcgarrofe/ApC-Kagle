##################################################
## score_model.py
## Score Model document for "Aprenentatge Computacional" - Kaggle project
##################################################
__author__ = "Marc Garrofé Urrutia"
__license__ = "MIT"
__version__ = "1.0.1"
__date__ = "11/10/2021"
##################################################

import numpy as np
from sklearn.model_selection import train_test_split
import time
from generate_features import standaritzador
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt

SPLIT_RATIO = 0.2
PLOT = False
SAVE_PLOTS = False

def printModelScore(model, X_train, y_train, X_test, y_test):
    """
    Given a models and the train and test data, prints the model score and MSE in training and test
    :param model: Model
    :param X_train: Input data train
    :param y_train: Outuput data train
    :param X_test: Input data test
    :param y_test: Outuput data test
    """
    print ('Training Score:', model.score(X_train, y_train) )
    print ('Testing Score:', model.score(X_test, y_test) )
    print ('Training MSE: ', np.mean((model.predict(X_train) - y_train)**2))
    print ('Testing MSE: ', np.mean((model.predict(X_test) - y_test)**2))

def validateModel(model, model_name, dataset, standaritze=False):
    """
    Given a model and a dataset, prints the elapsed time for fiting and the score of the model
    :param model_name: String amb el nom o descripció del model
    :param model: Model to be tested
    :param dataset: Fataframe object with the dataset data
    :param standaritze: Boleà que estandaritza les dades en cas que aquest sigui True
    :return model
    """
    x = dataset.drop('Class', axis=1).values # Guardem dades d'entrada
    if standaritze:
        x = standaritzador(x)
    y = dataset.values[:,0] # Guardem dades sortida
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=SPLIT_RATIO)
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    print()
    print(model_name, ' - Time: ', end - start)
    print ('Testing Score:', model.score(X_test, y_test) )
    print ('Testing MSE: ', np.mean((model.predict(X_test) - y_test)**2))
    return model

def plotModelInfo(model, dataset, plot_title, standaritze=False, confussion=False, roc=False, recall=False):
    """
    Plot the info of the model
    :param model: Model to be tested
    :param dataset: Fataframe object with the dataset data
    :param plot_title: String with the title of the plot
    :param standaritze: Boleà que estandaritza les dades en cas que aquest sigui True
    :param confussion: Boleà mostra Confussion Matrix
    :param roc: Boleà mostra ROC-Curve Plot
    :param recall: Boleà mostra Recall-Curve Plot
    """
    x = dataset.drop('Class', axis=1).values # Guardem dades d'entrada
    if standaritze:
        x = standaritzador(x)
    y = dataset.values[:,0] # Guardem dades sortida
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=SPLIT_RATIO)

    if PLOT:
        if roc:
            metrics.plot_roc_curve(model, X_test, y_test)
        if recall:
            metrics.plot_precision_recall_curve(model, X_test, y_test)
        if confussion:
            plot_confusion_matrix(model, X_test, y_test)
            plt.title(plot_title)
    if SAVE_PLOTS:
        file_name = "confussion_matrix" + str(plot_title) + ".png"
        plt.savefig(file_name)