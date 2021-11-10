1. **Keras permet de créer des réseaux de neurones par couches. Il existe de nombreuses couches différentes comme par exemple les couches Dense. Dans une couche Dense, chaque neurone d'une couche est connecté à tous les neurones de la couche précédente. Le code suivant permet de créer une couche Dense qui contient 45 neurones. Modifier le code pour qu'il contienne 60 neurones.**

```python
from tensorflow.keras.layers import Dense


layer = Dense(units=60)
```

2. **La plupart des neurones ont ce qu'on appelle une fonction d'activation. Il s'agit d'une fonction non linéaire qui modifie la sortie. En Keras on peut spécifier la fonction d'activation de la couche grâce au paramètre  "activation".Créer une couche Dense avec 60 neurones dont la fonction d'activation est "relu"**

```python
from tensorflow.keras.layers import Dense


Dense(units=60, activation='relu')
```

3. **Il existe deux manières de créer des réseaux de neurones en Keras : l'API dite séquentielle et l'API dite fonctionnelle. Dans cet exercice on s'intéresse à  l'API séquentielle. Le code ci-dessous permet de créer un réseau de neurone à deux couches. Modifier le réseau pour que**

    -**la première couche contienne 30 neurones et utilise la fonction d’activation relu**

    -**La seconde couche contienne 1 neurone et utilise la fonction d'activation linear**

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(units=30, activation='relu'))
model.add(Dense(units=1, activation='linear'))
```

4. **On a créé un réseau de neurone simple qui est stocké dans la variable model. Compilez le avec l'optimiseur “sgd" et la fonction de coût "mse". Il est possible de mettre ces arguments sous forme de chaîne de caractère.**

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(units=30, activation='relu'))
model.add(Dense(units=1, activation='linear'))
model.compile(optimizer='sgd', loss='mse')
```

5. **On dispose de données simple contenant le prix de maison en fonction de leur caractéristiques.** 

   1. **Importer le fichier csv houses.csv avec pandas**

   2. **Faire un train_test_split de votre dataframe**

   3. **Créer un modèle avec deux couches et compiler le modèle avec la bonne loss et l'optimiser “sgd”**

   4. **Créer une variable X_train contenant les colonne size et nb_room**

   5. **Créer une variable y_train**

   6. **Entraîner votre modèle avec la fonction fit de votre modèle. On prendra epochs=1**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('houses.csv')
X = df.iloc[:, :2].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


model = Sequential()
model.add(Dense(units=30, activation='relu'))
model.add(Dense(units=1, activation='linear'))
model.compile(optimizer='sgd', loss='mse')

model.fit(X_train, y_train, epochs=1)
```

6. **Expliquer à quoi sert une couche de neurone ayant une fonction d'activiation softmax. Dans quels cas est-ce utilisé ?**

Cette couche permet d'attribuer une classe à l'input dans le cadre d'un problème de classification à plus de 2 classes cibles.

7. **Créer une couche dense à l'aide de keras qui contiendrait 10 neurones et aura pour fonction d'activation  "softmax". Il n'y a pas besoin de créer de réseau, juste une simple couche**

```python
from tensorflow.keras.layers import Dense


layer = Dense(units=10, activation='softmax')
```

8. **On veut faire un petite réseau de neurone pour faire de la classification de tumeur (bénin / malin) à partir de deux caractéristiques.**

**Créer un réseau de neurone avec :** 

   1. **Une première couche contenant 20 neurones et avec la fonction d'activation relu. Il faudra spécifier l'argument input_shape.**

   2. **Une seconde couche avec fonction d'activation softmax (quel est le bon nombre de neurones ?)**

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(units=20, activation='relu', input_shape=[2]))
model.add(Dense(units=2, activation='softmax'))
```

9. **On veut faire de la classification de tumeur (bénin ou cancéreux) à partir de deux caractéristique (la taille et la concentration en protéine p53).** 

   1. **Avec pandas, charger le fichier tumors.csv dans un dataframe nommé df**

   2. **Faire un train_test_split des données**

   3. **Créer une variable X_train avec les colonne size et p53_concentration et une variable y_train**

   4. **Créer un réseaux de neurones avec trois couches. Les deux premières couches auront 30 et 15 neurones respectivement.**

   5. **Compiler le modèle avec la loss adéquate et l'optimiser sgd**

   6. **Entraîner le modèle en mettant epochs= 3**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('tumors.csv')
X = df.iloc[:, :2].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


model = Sequential()
model.add(Dense(units=30, activation='relu', input_shape=[2]))
model.add(Dense(units=15, activation='linear'))
model.add(Dense(units=2, activation='softmax'))
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')

model.fit(X_train, y_train, epochs=3)
```

10. **Comment faut-il choisir le learning rate pour que le modèle apprenne ? Pourquoi il ne faut pas qu'il soit trop gros ? Que se passe-t-il si il est trop petit ?**

    1. Il faut choisir le learning rate permettant de minimiser la fonction de coût (la loss) et permettant de garder ou d'augmenter la vitesse d'entraînement du modèle.

    2. Un trop grand learning rate alors il n'y aura que très peu de convergence / divergence et les changements de poids peuvent être trop grands pour l'optimiseur qui risque de faire augmenter la loss.

    3. Un learning rate trop petit coûtera davantage en termes d'optimisation parce que les étapes entre chaque diminution de la loss seront trop petites

11. **Explique ce qu'est un batch et ce qu'est une epoch.Prenez bien soin d'expliquer la différence entre les deux**

    - La taille du batch (lot) est un hyper-paramètre du gradient descent (méthode d'optimisation) qui contrôle le nombre d'entraînements à lancer avant que les paramètres internes du modèle ne soient mis à jour

    - Le nombre d'epochs (époques) est aussi un hyper-paramètre du gradient descent qui contrôle le nombre de passages à faire à travers le dataset.