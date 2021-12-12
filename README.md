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
Per tal de fer una prova, es pot fer servir amb la següent comanda:

``` python3 demo/demo.py ModelFileName.sav --input here ```

On:
* ModelFileName.sav indica el nom del arxiu que conté el model.
* --input here indica la sequüència de 13 dades encadenades separades epr una ',' (coma).

Es poden trobar possibles valors d'entrada en el document adjunt a la carpeta demo. Aquest arxiu conté dades que el model no coneix.

Per exemple:

``` python3 demo/demo.py RandomForest.sav 0.197433471679688,23.2397373922177,4.82076108018409,0.057362692872771,24.9598454758206,634.433355996892,701.371212121212,0.207747109103062,0.04315886134068,0.370373572705534,10.3560606060606,0.86380276292994,7.45834073120019E-155 ```

Retorna una classificació '1' (TUMOR)

``` python3 demo/demo.py RandomForest.sav 20.4353485107422,1227.15143951165,35.0307213672748,0.066763238434175,2.14462530729276,4.88203433514707,161.158675496689,0.225930928275797,0.051044784351564,0.502712050930512,5.08312582781457,0.952749246962317,7.4583407312002E-155 ```

Retorna una classificació '0' (NO TUMOR)
## Conclusions
Concluïm que Random Forest és el model que millor classifica les imatges celebrals.
Tot i que Random Forest és el més lent, la diferència respecte els altres models és imperceptible i en l'àmbit que s'aplicaria no repercutiria doncs es busca la màxima precisió.

## Idees per treballar en un futur
Per un treball a futur, es poden definir diferents models que analitzessin tot tipus de proves mèdiques com ara analítiques, biopsies, ecografíes, etc. per tal de tractar o detectar amb antel.lació malalties.
## Llicencia
El projecte s’ha desenvolupat sota llicència MIT.