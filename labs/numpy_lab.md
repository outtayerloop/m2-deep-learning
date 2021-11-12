# Numpy review lab

1. **A l’aide la fonction np.array créer une variable a contenant un vecteur numpy avec les valeurs [1, 2, 3]**

```python
import numpy as np

a = np.array([1, 2, 3])
```

2. **A l’aide la fonction np.arange créer un vecteur contenant les valeurs de 0 à 19.9 (inclu) espacés de 0.1 et stocker le résultat dans la variable a.**

```python
import numpy as np

a = np.arange(0, 20, 0.1)
```

3. **Numpy est une librairie qui permet de faire des opérations sur les tableaux, matrices et vecteurs de manière efficace et sans utiliser de boucle. On a créé deux vecteurs v1 et v2. Faire la somme de v1 et v2, le stocker dans une variable v3 et l'afficher. Que se passe-t-il ?**

```python
import numpy as np
v1 = np.array([1,2,3,4,5])
v2 = np.array([2,3,4,5,6])

v3 = v1 + v2

## Addition de chaque element des 2 vecteurs 1 par 1.
```

4. **Avec le sous module de numpy random, créer un vecteur numpy v1 contenant 1000 nombre aléatoires distribués selon une loi normale de moyenne 0 et d'écart type 1.**

```python
import numpy as np

rng = np.random.default_rng()
v1 = rng.standard_normal(1000) # mean=0, stdev=1
```

5. **A l'aide de la fonction zeros (regarder sur internet comment elle marche) de numpy, créer une matrice 10 par 10 remplie de zéros et la stocker dans la variable m**

```python
import numpy as np

m = np.zeros((10, 10))
```

6. **On a créé une matrice A de dimensions 10 par 10. Redimensionner la matrice pour qu'elle fasse 20 lignes et 5 colonnes et stocker le résultat dans la variable m**

```python
import numpy as np
A = np.zeros((10, 10))

#reshape matrice A
m = A.reshape((20, 5))
```

7. **On a créé une matrice A. Avec la méthode mean fournie par numpy, calculer les moyennes ligne par ligne et colonne par colonne et les stocker dans les variables row_mean et column_mean.**

```python
import numpy as np

A = np.matrix(np.arange(12).reshape((3, 4)))
row_mean = A.mean(1)
column_mean = A.mean(0)
```

8. **On a créé un vecteur numpy et on l'a stocké dans la variable v1. Créer un vecteur de boolean numpy (qu'on appellera mask) qui contiendra True si la valeur est strictement positive et False sinon. A l'aide du mask. Récupérer les valeurs positives de v1 et les stocker dans la variable positive_v1.**

```python

```