##################################################
## generate_features.py
## Generate Features (preprocessing) for "Aprenentatge Computacional" - Kaggle project
##################################################
__author__ = "Marc Garrofé Urrutia"
__license__ = "MIT"
__version__ = "1.0.1"
__date__ = "11/10/2021"
##################################################

# Llista atributs menys rellevants
list_features = ['Mean', 'Variance', 'Coarseness', 'Contrast', 'Correlation', 'Dissimilarity', 'Kurtosis', 'Skewness']

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

def featureSelection(dataset, list_features):
    """
    Donada una llista dels atributs NO rellevants, els elimina del dataset
    :param dataset: Objecte DataFrame amb les dades del dataset
    :param list_features: Llista amb els labels de les columnes a eliminar
    :return: Dataset amb les columnes rebudes eliminades
    """
    return dataset.drop(list_features, axis=1)