# Imports :
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from matplotlib import pyplot as plt
import os
from IPython.display import Image, display
import seaborn as sns
from sklearn.datasets import make_regression

import scipy.stats

from sklearn import linear_model
from sklearn.pipeline import make_pipeline

# Constants :
IMG_DATASET_PATH = 'data/Img/Brain Tumor'
CSV_DATASET_PATH = 'data/Brain Tumor.csv'
IMG_RES_X = 240
IMG_RES_Y = 240


# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Funcio per a llegir dades en format csv
def load_dataset(path):
    """
    :param path: String of the path to be loaded in a dataset
    :return: Dataset loaded in a Dataframe
    """
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

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


from sklearn.model_selection import cross_val_score
def printCSV(model, X_train, y_train, cv=5):
    """
    Given a models and the train split data, prints de Cross Validation Score
    :param model: Model to test
    :param X_train:  Input data train
    :param y_train: Outuput data train
    """
    print('Cross Validation Score: ', np.mean(cross_val_score(model, X_train, y_train, cv=cv)))

from sklearn.preprocessing import StandardScaler
def standaritzador(data):
    """
    Given a DataFrame, standarizes all the columns
    :param data: DataFrame data
    :return: DataFrame data standarized
    """
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)


"""
def gridSearch(estimator, param_grid, model_name, X_train, y_train):
    
    Executes the GridSearchCV function ans shows the statistics
    :param estimator: Model object to be tested
    :param param_grid: Dict with the diferent values to be tested
    :param model_name: String with the title to be shown
    :param X_train: Fataframe with the input data
    :param y_train: Fataframe with the output data
    
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print(model_name)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
"""

def gridSearch(estimator, param_grid, model_name, dataset):
    """
    Executes the GridSearchCV function ans shows the statistics
    :param estimator: Model object to be tested
    :param param_grid: Dict with the diferent values to be tested
    :param model_name: String with the title to be shown
    :param dataset: DataFrame amb la informació i dades del dataset
    """
    x = dataset.values[:,1:-1] # Guardem dades d'entrada
    y = dataset.values[:,0] # Guardem dades sortida
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print(model_name)
    print(grid_search.best_params_)
    print(grid_search.best_score_)

def featureSelection(dataset, list_features):
    """
    Donada una llista dels atributs NO rellevants, els elimina del dataset
    :param dataset: Objecte DataFrame amb les dades del dataset
    :param list_features: Llista amb els labels de les columnes a eliminar
    :return: Dataset amb les columnes rebudes eliminades
    """
    return dataset.drop(list_features, axis=1)


# Carreguem dataset d'exemple
dataset = load_dataset(CSV_DATASET_PATH)


# Eliminem variables que no són rellevants o no tenen un impacte significatiu a l'hora de decidir la classe d'una imatge
dataset = dataset.drop(['Image'], axis=1)
# dataset = dataset.drop(['Mean', 'Variance', 'Coarseness', 'Contrast', 'Correlation', 'Dissimilarity', 'Kurtosis', 'Skewness'], axis=1)
# Guardem dades d'entrada
x = dataset.values[:,1:-1]
# Guardem dades sortida
y = dataset.values[:,0]


# Divisió Train i Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

param_grid = [
    {
        'penalty' : ['l2'],
        'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'warm_start' : ['True', 'False']
    },
    {
        'penalty' : ['l1'],
        'solver' : ['liblinear', 'saga'],
        'warm_start' : ['True', 'False']
    }
]


logistic_regressor = linear_model.LogisticRegression() # Definim model reg. logistic
gridSearch(logistic_regressor, param_grid, 'Logistic Regression', dataset)


logistic_regressor = linear_model.LogisticRegression()
gridSearch(logistic_regressor, param_grid, 'Logistic Regression', X_train, y_train)
# BEST CONFIGURATION = {'penalty': 'l1', 'solver': 'liblinear', 'warm_start': 'True'}


# Repetim el mateix proces pero fent Estandaritzacio:
logistic_regressor = linear_model.LogisticRegression()
X_train_scaled = standaritzador(X_train)
grid_search = GridSearchCV(estimator=logistic_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print("Logistic Regression + Estandaritzacio")
print(grid_search.best_params_)
print(grid_search.best_score_)



# Repetim el mateix proces pero fent Feature Selection:
dataset = dataset.drop(['Mean', 'Variance', 'Coarseness', 'Contrast', 'Correlation', 'Dissimilarity', 'Kurtosis', 'Skewness'], axis=1)
x = dataset.values[:,1:-1]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
grid_search = GridSearchCV(estimator=logistic_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
logistic_regressor = linear_model.LogisticRegression()
print("Logistic Regression + Feature election")
print(grid_search.best_params_)
print(grid_search.best_score_)



# Repetim el mateix proces pero fent Feature Selection + Estandaritzacio:
logistic_regressor = linear_model.LogisticRegression()
X_train = standaritzador(X_train)
grid_search = GridSearchCV(estimator=logistic_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print("Logistic Regression + Feature Selection + Estandaritzacio")
print(grid_search.best_params_)
print(grid_search.best_score_)




