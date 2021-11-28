# Deep learning avec keras II: CNN and MNIST

1. **Expliquer ce qu'est une convolution 2D en traitement d'image et le resultat d'une telle opération sur une image**

Une convolution 2D consiste en effectuer le produit de convolution de la matrice initiale 2D représentant l'image d'input avec une matrice de transformation appelée le kernel. Ce produit agit bloc par bloc sur la matrice d'origine et non pas pixel par pixel. L'image finale est constituée de chaque bloc initial transformé par le kernel, qui peut aboutir à une image floutée par exemple ou qui a subi une rotation

2. **Les réseaux de neurones convolutionnels comportent deux parties : la première qui contient les couches convolutionnelles et la seconde qui est composées de couches Denses. Le code ci-dessous permet de créer une couche convolutionnelle, néanmoins la taille de la convolution est trop grande.De plus, il contient un bug. Modifier le code pour que la taille soit de 3x3 et qu'il fonctionne**

```python
from tensorflow.keras.layers import Conv2D

conv = Conv2D(10, kernel_size=(3, 3), activation="relu")
```

3. **Les réseaux de neurones convolutionnels comportent deux parties : la première qui contient les couches convolutionnelles et la seconde qui est composées de couches Denses. Créer une couche convolutionnelle qui contient 30 neurones, un kernel_size de (3, 3) et un strides de 2x2**

```python
from tensorflow.keras.layers import Conv2D

conv = Conv2D(10, kernel_size=(3, 3), strides=(2, 2), activation="relu")
```

4. **La partie convolutionnelle d'un réseau peut contenir ce qu'on appelle des couches de Pooling qui permettent de réduire la taille des images. Créer une couche une couche de Max pooling 2D avec comme argument pool_size = (3,3)**

```python
from tensorflow.keras.layers import MaxPooling2D

conv = MaxPooling2D(pool_size=(3, 3))
```

5. **Expliquer ce que fait une couche de pooling.Expliquer les avantages et les inconvénients**

La couche de pooling va sous-échantillonner le bloc sélectionné pour le produit de convolution avec la couche convolutionnelle afin de réduire le nombre de neurones des couches suivantes et ainsi ne retenir que l'information la plus "importante". L'inconvénient peut être de perdre des informations qui pouvaient être nécessaires et ainsi d'avoir de moins bonnes performances de classification d'image dans les couches suivantes

6. **Les couches convolutionnelles 2D  on besoin de Tenseur 4D ! La première dimension représente toujours le nombre d'exemples. La seconde et la troisième représentent les dimensions de l'image. La quatrième représente le nombre de cannaux (1 pour noir et blanc, 3 pour les couleurs). Il y a quatre tableaux numpy chargés dans votre environnement: x_train et x_test (images) et y_train et y_test (labels correspondants aux chiffres affichées par les images).**

   1. **Afficher la taille de x_train et x_test**

   2. **Avec la méthode reshape, modifier la taille de x_train et x_test pour que toutes les images passe d'une taille (28, 28) à (28, 28, 1)**

```python
print(x_train.shape)
print(x_test.shape)
x_train = x_train.reshape(200, 28, 28, 1)
x_test = x_test.reshape(30, 28, 28, 1)
```

7. **Avec Keras créer un réseau de neurones avec les couches suivantes : Convolution avec filters=8 et kernel_size=(3, 3). Flatten. Dense avec 10 neurones et une  activation softmax Spécifier l'argument input_shape en supposant qu'on a des images en noir et blanc de taille 28 par 28. Appelez votre modèle model**

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D

model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3), activation="relu", input_shape=[28, 28, 1]))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))
```

8. **On a créé un réseau de neurone convolutionnel qui est stocké dans la variable   model. Compilez le avec l'optimiseur Adam, la fonction de coût sparse_categorical_crossentropy et la métrique accuracy.**

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

9. **On a chargé un CNN dans la variable model et on l'a compilé. Les variables model, x_train, x_test, y_train et y_test sont toutes chargées dans votre environnement. Entrainez le modèle avec 6 époques (epoch). Il faudra passer les données de validation à la fonction d'entraînement.**

```python
model.fit(x_train, y_train, validation_data= (x_test, y_test), epochs=6)
```

10. Expliquer la différence entre une couche Dense et convolutionnelle

Une couche de convolution permet d'extraire des blocs de la matrice contenant les pixels de l'image d'origine en réalisant un produit de convolution sur chaque bloc avec une matrice appelée le kernel. Elle permet d'extraire des informations visuelles de l'image afin de les passer par la suite aux couches denses qui vont elles s'occuper de la classification d'image à proprement parler