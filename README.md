# Pràctica Kaggle APC UAB 2021-22
### Nom: Marc Garrofé Urrutia
### Niu: 1565644
### DATASET: Brain Tumor
### URL: [kaggle](https://www.kaggle.com/jakeshbohaju/brain-tumor)
## Resum
El dataset utilitza dades obtingudes a partir de l'anàlisi d'imatges celebrals.
Tenim 3762 filas de dades amb 15 atributs o columnes. Totes les dades que fan referència a l'imatge són numèriques (13 sobre 15 columnes). Les dues columnes restants, fan referència a la classificació final de la imatge (atribut bianri o dummy) i l'altre és un dada identificadora.
### Objectius del dataset
Volem aprender quin és el millor model que prediu si una imatge celebral conté un tumor o no.
## Experiments
Durant aquesta pràctica he realitzat diferents execucions amb múltiples models, tals com:
* Regrsssor Logístic
* Gradient Descent
* Suported Vector Classfication
* Random Forest
* Naive Bayes
### Preprocessat
El dataset proporcionat no conté valors Nulls i aquest està balancejat (55,26% són classe 0 i el 44,74% són de la classe 1).
Per a diferents execucions aplicant Feature Selection, el model no millora en accuracy i en la majoria de casos hi ha una diferencia d'empitjorament entre el 1% i el 3%.
### Model
A continuació mostrem els millors resultats d'accuracy obtinguts amde cada model aplicant  el Grid Search i CV:
| Model | Preprocessing | Hiperparametres | Mètrica | Temps |
| -- | -- | -- | -- | -- |
| Regresor Logístic | Estandarització | penalty: l2, solver: newton-cg, warm_start: True | 98.30 % | 1.87 s |
| Gradient Descent | Estandarització | 'alpha': 0.001, 'class_weight': 'balanced', 'fit_intercept': True, 'learning_rate': 'optimal', 'loss': 'log', 'max_iter': 5000, 'penalty': 'l2', 'shuffle': True, 'warm_start': 'True' | 98.24 % | 4.78 s |
| SVC | Estandarització | 'class_weight': 'balanced', 'gamma': 'scale', 'kernel': 'linear' | 98.30 % | 1.37 s |
| Random Forest | default | 'balanced_subsample', 'criterion': 'gini', 'max_features': 'auto', 'n_estimators': 75, 'warm_start': True | 98.91 % | 101.31 s |
| Naive Bayes | Feature Selection | default | 97.25 % | 0.01 s |
Aquells valors marcats com a 'default' signifiquen que no s'ha realitzat cap tipus de modificació. Per exemple, 'default' en preprocessing indica que no s'han tractat les dades i en hipermaràmetres inica que s'ha utilitzat la configuració per defecte del model.
## Demo
Per tal de fer una prova, es pot fer servir amb la següent comanda
``` python3 demo/demo.py --input here ```
## Conclusions
El millor model que s'ha aconseguit ha estat Random Forest
En comparació amb l'estat de l'art i els altres treballs que hem analitzat....
## Idees per treballar en un futur
Crec que seria interesant indagar més en
## Llicencia
El projecte s’ha desenvolupat sota llicència MIT.