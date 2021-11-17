# Linear Regression review lab

2. **Vous êtes data scientist dans une agence immobilière MODERNE et à la pointe de la technologie. On vous demande de faire un modèle afin de prédire le prix d'une maison en fonction de ses caractéristiques (taille, nombre de chambre, etc.) . Nous allons commencer par découvrir le dataset.**

   1. **Charger le dataset houses.csv avec pandas dans le dataframe df.**

   2. **Créer un variable columns contenant les colonnes du dataset grâce à l'attribut .columns, puis les afficher.**

   3. **Afficher les premières lignes du dataset avec la méthode head.**

   4. **Afficher le scatter plot du prix en fonction de la taille.**

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('houses.csv')
columns = df.columns
print(columns)
df.head()
plt.scatter(df['size'], df['price'])
```

3. **Après avoir découvert notre dataset, nous allons pouvoir entrainer un premier modèle. Grâce à la taille et au nombre de chambre, nous allons prédire le prix d'une maison.** 

   1. **Charger le dataset houses.csv avec pandas dans une variable df.**

   2. **Créer une variable X qui contient les données des colonnes 'size' et 'nb_rooms'.**

   3. **Créer une variable y qui contient la colonne 'price'.**

   4. **Créer une régression linéaire avec scikit-learn et la stocker dans la variable  model. Entraîner le modèle avec la méthode fit de votre modèle.**

   5. **Calculer la mse du modèle avec la fonction mean_squared_error de scikit-learn et stocker le resultat dans une variable nommée mse**

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('houses.csv')
X = df[['size', 'nb_rooms']]
y = df['price']
model = LinearRegression()
model = model.fit(X, y)
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
```

4. **On a fait notre première régression. Bien entendu c'est loin d'être parfait et on va voir comment l'améliorer. Pour cela on va essayer d'utiliser plus de variables. Comme la colonne Orientation. Malheureusement elle est en "string" et on ne peut pas l'utiliser dans le modèle. Il faut la convertir en colonne numérique.** 

   1. **Charger le dataset houses.csv avec pandas dans le dataframe df.**

   2. **Avec la classe LabelEncoder de sklearn transformer en données numérique les données de la colonne orientation.**

   3. **Faire un modèle qui utilise la colonne numérique orientation et les colonnes size et nb_rooms.**

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

df = pd.read_csv('houses.csv')
le = LabelEncoder()
df['orientation'] = le.fit_transform(df['orientation'])
model = LinearRegression()
X_train = df[['orientation', 'size', 'nb_rooms']]
y_train = df['price']
model = model.fit(X_train, y_train)
```

5. **Le label encoder nous permet de transformer une variable catégorielle sous forme de texte en nombre 0, 1, 2, 3 mais le problème c'est que ça induit une relation hierarchique entre les catégories. Ou des fois il n'y a pas de relation naturelle. Dans ce cas on peut utiliser le one hot encoding.**

   1. **Charger les données dans un dataframe et stocker dans la variable df.**

   2. **A l'aide la méthode pd.get_dummies one_hot encoder la colonne orientation et faire sorte que les nouvelles colonnes remplace la colonne orientation**.

   3. **Entraîner un modèle avec les colonnes orientations , size, nb_rooms et garden.**

   4. **Afficher la variable mse du modèle.**

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('houses.csv')
dummies = pd.get_dummies(df['orientation'])
df = pd.concat([df, dummies], axis=1)
model = LinearRegression()
X_train = df[['Est', 'Nord', 'Ouest', 'Sud', 'size', 'nb_rooms', 'garden']]
y_train = df['price']
model = model.fit(X_train, y_train)
y_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
```

6. **Afin de vérifier qu'un modèle généralise bien on sépare généralement notre dataset en deux : un jeux de données de train et un jeu de données de test. Le test est gardé secret pour évaluer le modèle après l'entraînement.**

   1. **Charger le dataset "houses.csv" dans un dataframe pandas nommé df.**

   2. **Avec la fonction train_test_split du module model_selection de scikit learn, séparer dataframe pandas en une partie train et une partie test. Stocker la partie train dans la variable train et la partie test dans la variable test.**

   3. **Afficher l'attribut  shape du jeu de train et test généré à l'aide de la fonction print.**

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('houses.csv')
train, test = train_test_split(df)
print(train.shape)
print(test.shape)
```

7. **Maintenant que vous savez séparer votre dataset entre train et test, vous allez pouvoir entrainer et tester une régression linéaire à l'aide de ces données.**

   1. **charger le dataset dans un dataframe pandas.**

   2. **Reprendre le traitement des données et le faire sur le train et le test (orientation).**

   3. **Faire le train test split de votre dataframe et entraîner votre modèle sur le dataset de train.**

   4. **Cacluler la MSE de votre model sur les données de train et stocker le résultat dans la variable mse_train**

   5.  **Cacluler la MSE de votre model sur les données de test et stocker le résultat dans la variable mse_train**

   6. **Stockez les coefficients du modèle dans la variable model_coefs.**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

df = pd.read_csv('houses.csv')
dummies = pd.get_dummies(df['orientation'])
df = pd.concat([df, dummies], axis=1)
model = LinearRegression()
X = df[['Est', 'Nord', 'Ouest', 'Sud', 'size', 'nb_rooms', 'garden']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X,y)
model = model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
mse_train = mean_squared_error(y_train, y_pred_train)
y_pred_test = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
model_coefs = model.coef_
```

8. **On souhaite prédire, non pas une valeur continue comme le prix d'une maison mais si une tumeur est cancéreuse (y = 1y=1) ou non (y=0) à partir de caractéristiques. Peut-on utiliser la régression linéaire pour cela ? Cela peut-il poser des problèmes ?**

Non car la régression linéaire produit une droite dont les y seront la prédiction. Pour une classification binaire 0, 1, cela pose problème car les points seront éloignés de la droite et la prédiction sera donc éloignée de la réalité (surtout qu'elle ne risque pas d'être à 0 ou à 1 tout le temps si c'est une droite de type ax + b)

9. **Quand on utilise la méthode .score pour une régression linéaire, cela retourne le coefficient R2. Qu'est ce que le coefficient R2 ? que mesure-t-il ?**

R2 (R squared) mesure la qualité de prédiction de la régression linéaire, c'est une métrique d'évaluation du modèle


