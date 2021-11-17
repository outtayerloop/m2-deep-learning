# Logistic Regression review lab

1. **On souhaite prédire, non pas une valeur continue comme le prix d'une maison mais si une tumeur est cancéreuse (y = 1y=1) ou non (y=0) à partir de caractéristiques.Peut-on utiliser la régression linéaire pour cela ? Cela peut-il poser des problèmes ?**

Non car la régression linéaire produit une droite dont les y seront la prédiction. Pour une classification binaire 0, 1, cela pose problème car les points seront éloignés de la droite et la prédiction sera donc éloignée de la réalité (surtout qu'elle ne risque pas d'être à 0 ou à 1 tout le temps si c'est une droite de type ax + b)

2. **La régression logistique est un modèle qui permet de faire de la classification. Expliquer les ressemblances et liens entre régression logistique et régression linéaire**

La régression logistique reste une combinaison linéaire de paramètres teta comme la régression linéaire sauf qu'on lui applique une fonction non linéaire qui est la fonction sigmoïde.

3. **Charger le fichier  tumor_data_one_var.csv qui contient des données sur des tumeurs bénignes et cancereuses. Le fichier contient deux colonnes : size et is_cancerous.**

   1. **Créer une instance de régression logistique avec scikit learn et la stocker dans la variable model.**

   2. **Entraîner la régression logistique sur les données : y = f0(size)**

   3. **Afficher les paramètres du modèles**

   4. **Afficher le score du modèle avec la méthode score de votre modèle.**

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('tumor_data_one_var.csv')
model = LogisticRegression()
X = df[['size']]
y = df['is_cancerous']
model = model.fit(X, y)
print(model.coef_)
print(model.score(X,y))
```

4. **Charger le fichier  tumor_data_two_var.csv dans un dataframe pandas et stocker la variable dans la variable df.**

   1. **Faire un train test split.**

   2. **Créer une instance de la classe LogisticRegression et la stocker dans une variable nommée "model". Entraîner le modèle sur le dataset de train. Le modèle prédira si la tumeur est cancereuse ou non à partir des variables du dataframe. Modele de la forme y = f0(size, concentration)**

    3. **Créer une instance de la classe LogisticRegression et la stocker dans une variable nommée "model". Entraîner le modèle sur le dataset de train. Le modèle prédira si la tumeur est cancereuse ou non à partir des variables du dataframe.**

    4. **Afficher le score du modèle sur le jeu d'entraînement et sur le jeu de test. est-ce qu'il semble marcher ? Sinon, que proposez-vous comme solution ?** 

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('tumor_data_one_var.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X,y)
model = LogisticRegression()
model = model.fit(X_train, y_train)
print(model.score(X_train,y_train))
model = model.fit(X_test, y_test)
print(model.score(X_test,y_test))
# 83% d'accuracy sur dataset de test : 
# pas mauvais mais peut s'améliorer
# en normalisant les features.
```

5. Charger le fichier  tumor_data_two_var.csv. A l'aide de la classe MinMax Scaler, normaliser les colonnes size et p53_concentration.

    1. Avec scikit learn, entraîner une régression logistique qui prédit si la tumeur est cancereuse ou non à partir des variables du dataframe. Stocker le resultat dans model. Modele de la forme y = f0(size, concentration)

    2. Afficher le score du modèle. Cela marche-t-il mieux ? 

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('tumor_data_two_var.csv')
sc = MinMaxScaler()
df[['size', 'p53_concentration']] = sc.fit_transform(df[['size', 'p53_concentration']])
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X,y)
model = LogisticRegression()
model = model.fit(X_train, y_train)
print(model.score(X_train,y_train))
model = model.fit(X_test, y_test)
print(model.score(X_test,y_test))
# Accu de 96% : largement mieux avec MinMaxScaler
```

6. **Charger le fichier  tumor_data_two_var.csv. Sans normaliser les données, entraîner un arbre de décision. Afficher le score sur le train et sur le test. Afficher la matrice de confusion sur le train et sur le test.**

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv('tumor_data_two_var.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X,y)
model = DecisionTreeClassifier()
model = model.fit(X_train, y_train)
print(model.score(X_train,y_train))
y_pred_train = model.predict(X_train)
print(confusion_matrix(y_train, y_pred_train))
model = model.fit(X_test, y_test)
y_pred_test = model.predict(X_test)
print(confusion_matrix(y_test, y_pred_test))
print(model.score(X_test,y_test))
```

7. **Donner l'équation de la fonction sigmoid**

```
f(x) = 1 / (1 = e^-x)
```

8. **On souhaite faire un modèle qui permet de prédire si une personne a un risque cardiaque en fonction de son poids et son age. Donner l'équation de la régression logistique avec les variables suivantes : theta0, theta1, theta2, age et poids. La variable age sera la première variable et poids la seconde.**

```
Soit f(x) = 1 / (1 = e^-x) la fonction sigma.
y = sigma(theta0 + theta1 * age +  theta2 * poids)
```