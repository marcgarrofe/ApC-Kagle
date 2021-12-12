##################################################
## train_model.py
## Train Model document for "Aprenentatge Computacional" - Kaggle project
##################################################
__author__ = "Marc Garrofé Urrutia"
__license__ = "MIT"
__version__ = "1.0.1"
__date__ = "11/10/2021"
##################################################

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import time
from generate_features import standaritzador

SPLIT_RATIO = 0.2

def gridSearch(estimator, param_grid, model_name, dataset, standaritze=False):
    """
    Executes the GridSearchCV function ans shows the statistics
    :param estimator: Model object to be tested
    :param param_grid: Dict with the diferent values to be tested
    :param model_name: String with the title to be shown
    :param dataset: DataFrame amb la informació i dades del dataset
    :param standarize: Boleà que estandaritza les dades d'entrada en cas de ser True
    """
    x = dataset.drop('Class', axis=1).values
    if standaritze:
        x = standaritzador(x)
    y = dataset.values[:,0] # Guardem dades sortida
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=SPLIT_RATIO)
    start = time.time()
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    end = time.time()
    print(model_name + " Time : " + str(end - start) )
    print(grid_search.best_params_)
    print(grid_search.best_score_)