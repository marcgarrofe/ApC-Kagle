

import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay

def plotLogisticRegression(model, X_test, y_test):
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.show()

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

def plotPrecisionRecallDisplay(model, y_test, y_score):
    prec, recall, _ = precision_recall_curve(y_test, y_score, pos_label=model.classes_[1])
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()


def modelLogisticRegression(X_train, y_train):
    print('\nRegressor Logistic + Standarització\n')
    model = make_pipeline(StandardScaler(), linear_model.LogisticRegression())
    model.fit(X_train, y_train)
    return model


from sklearn.linear_model import SGDClassifier

def modelSGDC(X_train, y_train, max_iter=1000):
    print ('\nGradient Descent\n')
    model = SGDClassifier(loss='log', penalty="l2", max_iter=max_iter)
    model.fit(X_train, y_train)
    SGDClassifier(max_iter=10)      # que es aixo?
    return model

def modelRegressorLogistic(X_train, y_train):
    print ('\nRegressor Logistic\n')
    model = linear_model.LogisticRegression()
    model.fit(X_train, y_train)


from sklearn import preprocessing

def modelGradientDescent(X_train, y_train, stantaritzation=True):
    model = SGDClassifier(loss='log', penalty="l2", max_iter=1000)
    if(stantaritzation):
        print ('\nGradient Descent + Standarització\n')
        scaler = preprocessing.StandardScaler().fit(X_train)
    else:
        print ('\nGradient Descent\n')
    model.fit(X_train, y_train)
    SGDClassifier(max_iter=10)
    return model



from sklearn import svm
def SVC(X_train, y_train, kernel):
    print ('\nSVC Linear\n')
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    printModelScore(clf, X_train, y_train, X_test, y_test)


print ('\nSVC Sigmoid\n')
clf = svm.SVC(kernel='sigmoid', C=1).fit(X_train, y_train)
printModelScore(clf, X_train, y_train, X_test, y_test)

from sklearn.ensemble import BaggingClassifier


from sklearn.ensemble import RandomForestRegressor

print ('\nRandom Forest\n')
regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
printModelScore(regr, X_train, y_train, X_test, y_test)






dataset = load_dataset(CSV_DATASET_PATH)
# Eliminem variables que no són rellevants o no tenen un impacte significatiu a l'hora de decidir la classe d'una imatge
dataset = dataset.drop(['Image'], axis=1)
# Guardem dades d'entrada
x = dataset.values[:,1:-1]
# Guardem dades sortida
y = dataset.values[:,0]

# Divisió Train i Test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


print ('\nRegressor Logistic + Standarització\n')
model = make_pipeline(StandardScaler(), linear_model.LogisticRegression())
model.fit(X_train, y_train)
printModelScore(model, X_train, y_train, X_test, y_test)


from sklearn.linear_model import SGDClassifier

print ('\nGradient Descent\n')
model = SGDClassifier(loss='log', penalty="l2", max_iter=1000)
model.fit(X_train, y_train)
SGDClassifier(max_iter=10)
printModelScore(model, X_train, y_train, X_test, y_test)

print ('\nRegressor Logistic\n')
model = linear_model.LogisticRegression()
model.fit(X_train, y_train)
printModelScore(model, X_train, y_train, X_test, y_test)



from sklearn import preprocessing

print ('\nGradient Descent + Standarització\n')
model = SGDClassifier(loss='log', penalty="l2", max_iter=1000)
scaler = preprocessing.StandardScaler().fit(X_train)
model.fit(X_train, y_train)
SGDClassifier(max_iter=10)
printModelScore(model, X_train, y_train, X_test, y_test)

from sklearn import svm

print ('\nSVC Linear\n')
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
printModelScore(clf, X_train, y_train, X_test, y_test)



print ('\nSVC Sigmoid\n')
clf = svm.SVC(kernel='sigmoid', C=1).fit(X_train, y_train)
printModelScore(clf, X_train, y_train, X_test, y_test)


from sklearn.ensemble import RandomForestRegressor

print ('\nRandom Forest\n')
regr = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
printModelScore(regr, X_train, y_train, X_test, y_test)




