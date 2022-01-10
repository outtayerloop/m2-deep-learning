# Deep Learning with Pytorch I : tensors and operations

1. **Créer un tensor t1 avec les valeurs [1, 2, 3]. Avec la méthode numpy() convertir t1 en tableau numpy et le stocker dans la variable np_t1**

```python
import torch



t1 = torch.tensor([1,2,3])
np_t1 = t1.numpy()
```

2. **Pytorch est une librairie de deep learning qui a de nombreuses similarités avec numpy : permet la création de tenseurs, fournit des opérations mathématiques associées bien sur, c'est beaucoup plus que cela. Avec la fonction torch.Tensor, créer un tenseur contenant les valeur [[1, -1], [0, 1.2]] et le stocker dans la variable t1. Afficher le type du tenseur. Créer un tenseur de type double avec les même valeur et les stocker dans la variable t2**

```python
import torch


t1 = torch.tensor([[1, -1], [0, 1.2]])
print(t1.type())
t2 = torch.tensor([[1, -1], [0, 1.2]], dtype=torch.double)
```

3. **Pytorch permet d'effectuer des opérations mathématiques comme numpy. Créer deux tensor Float t1 et t2 avec les valeurs [0.5, 1.5] et [-1.5, -0.5]. Calculer t1 + t2 et le stocker dans t3. Calculer t1 * t2 et le stocker dans t4. Cacluler l'exponentielle de t1 avec la fonction torch.exp et stocker la valeur dans t5.**

```python
import torch


t1 = torch.tensor([0.5, 1.5], dtype=torch.float)
t2 = torch.tensor([-1.5, -0.5], dtype=torch.float)
t3 = t1 + t2
t4 = t1 * t2
t5 = torch.exp(t1)
```

4. **Créer un tenseur avec 100 valeurs aléatoires distribuée aléatoirement et le stocker dans la variable t1. Calculer la somme des valeur avec torch.sum et stocker le resultat dans sum_t1. Calculer le maximum des valeurs avec torch.max et stocker le résultat dans max_t1. Calculer la norme de t1 et stocker le résultat dans norm_t1.**

```python
import torch


t1 = torch.rand(100)
sum_t1 = torch.sum(t1)
max_t1 = torch.max(t1)
norm_t1 = torch.norm(t1)
```

5. **Faire une fonction pytorch_relu(vector) qui permet de calculer le relu de vector**

```python
import torch

def pytorch_relu(vector):
    m = torch.nn.ReLU(inplace=True)
    m(vector)
    return vector
```

6. **Faire une fonction pytorch_softmax(activation_tensors) qui transforme le tensor 1D activation_tensors via la fonction softmax**

```python
import torch


def pytorch_softmax(activation_tensors):
    m = torch.nn.Softmax()
    output = m(activation_tensors)
    return output
```

7. **Faire une fonction pytorch_sigmoid(activation_tensors) qui calcule la sigmoid pour chaque valeur d'un tensor torch**

```python
import torch


def pytorch_sigmoid(activation_tensors):
    m = torch.nn.Sigmoid()
    output = m(activation_tensors)
    return output
```

8. **Faire une fonction linear_regression_prediction(thetas, x) qui calcule la sortie d'une régression linéaire.On supposera que x[0] = 1 pour que x et thetas soient de même taille**

```python
import torch


def linear_regression_prediction(thetas, x):
    print(x)
    print(thetas)
    m = torch.nn.Linear(x.shape[0], 1, bias=False)
    m.weight.data = thetas
    return m(x)
```