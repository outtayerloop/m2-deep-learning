# Deep Learning with Pytorch II : automatic differenciation and first models

1. **Créer un tenseur de type Float à une dimension contenant la valeur -1 et la stocker dans la variable t. Il faudra que le tensor ait la propriété requires_grad = True. Calculer exp(−t²) et stocker le résultat dans la variable z. Appeler la méthode .backward sur le tenseur z. Récupérer le gradient de z par rapport à t (utiliser l'attribut .grad) et le stocker dans la variable grad_t.**

```python
import torch


t = torch.tensor([-1], dtype=torch.float, requires_grad=True)
z = torch.exp(-t*t)
z.retain_grad()
z.backward()
grad_t = t.grad
```

2. **Créer trois tenseurs x, y et z contenant respectivement les valeurs 2.0, -2.0 et 1.0. En plus de spécifier la valeur vous ajouterez l'argument requires_grad=True. Calculer exp(x * z + y^2)exp(x∗z+y²) et stocker le résultat dans la variable result. Appeler la méthode backward de la variable result. Afficher la dérivée partielle de result par rapport à x, y et z.**

```python
import torch


x = torch.tensor([2.0], dtype=torch.float, requires_grad=True)
y = torch.tensor([-2.0], dtype=torch.float, requires_grad=True)
z = torch.tensor([1.0], dtype=torch.float, requires_grad=True)
result = torch.exp(x * z + y * y)
result.retain_grad()
result.backward()

# Dérivée partielle de result par rapport à x
print(x.grad)

# Dérivée partielle de result par rapport à y
print(y.grad)

# Dérivée partielle de result par rapport à z
print(z.grad)
```

3. **Dans cet exercice on va créer notre premier modèle. On a créé un caneva de code qui permet de créer un modèle Compléter le code de la classe LinearRegression. Créer un objet du type LinearRegression et le stocker dans la variable model en prenant n_feature  = 2. Calculer la prédictions du modèle et stocker le résultat dans la variable y_pred pour X valant (1, 1). Grâce à l'héritage, la classe LinearRegression dispose d'une méthode parameters qui permet de retourner tous les paramètres du model sous forme de générateur. A l'aide d'une boucle faire afficher l'ensemble des valeurs des paramètres de votre modèle.**

```python
import torch
import torch.nn as nn


class LinearRegression(nn.Module):
    
    def __init__(self, n_feature):
        super(LinearRegression, self).__init__()
        self.regression = nn.Linear(n_feature, 1)
        
        
    def forward(self, x):
        return self.regression(x)

model = LinearRegression(2)

x = torch.tensor([[1, 1]], dtype=torch.float)
y_pred = model.forward(x)

parameters = model.parameters()
for param in parameters:
    print(param)
```

4. **Créer une fonction mse(y, y_pred) qui calcule la mse : (1/m) * somme de i = 0 à n ((yi - y_predi)²). Créer un tenseur y de type Float valant  [0.5, -0.5, 1, 0]. Créer un tenseur x de type Float valant  [[1], [-1], [1], [1]]. Créer un modèle issue de la classe LinearRegression et le stocker dans la variable model. Calculer y_pred avec votre modèle. Avec torch.nn.MSELoss Calculer, la mse de votre modèle entre y_pred et et y. Stocker la variable dans mse1. Utiliser la méthode backward sur la variable mse1. Afficher les dérivées partielles de mse par rapport à chaque paramètre du modèle.**

```python
import torch
import torch.nn as nn
import numpy as np


class LinearRegression(nn.Module):
    
    def __init__(self, n_feature):
        super(LinearRegression, self).__init__()
        self.regression = nn.Linear(n_feature, 1)
        
        
    def forward(self, x):
        return torch.flatten(self.regression(x))
        
    
    def mse(self, y, y_pred):
        m = len(y)
        return (1/m) * np.sum((y - y_pred) ** 2)
        
        
y = torch.tensor([0.5, -0.5, 1, 0], dtype=torch.float)
x = torch.tensor([[1], [-1], [1], [1]], dtype=torch.float)

model = LinearRegression(x.shape[1])
y_pred = model.forward(x)

loss = torch.nn.MSELoss()
mse1 = loss(y_pred, y)
self_mse = model.mse(y.numpy(), y_pred.detach().numpy())
print(mse1.detach().numpy()) # pytorch mse
print(self_mse) # self coded mse
mse1.backward() # computes the partial derivatives of loss 
# regarding 

#Affiche les dérivées partielles de mse 
# par rapport à chaque paramètre du modèle.
parameters = model.parameters()
for param in parameters:
    print(param.grad) # derivee partielle de mse1 par rapport
    # au param
```

5. **En se basant sur l'exercice LinearRegression, Coder une classe LogisticRegression permettant de faire une regression logistique avec un nombre quelconque de feature. Il faudra donc implémenter les méthodes __init__ et forward.  Instancier un modèle dans une variable model acceptant 3 features. Calculer la prediction du modèle pour X = 0, 0, 0 et stocker la valeur dans y_pred.**

```python
import torch
import torch.nn as nn
import numpy as np


class LogisticRegression(nn.Module):
    
    def __init__(self, n_feature):
        super(LogisticRegression, self).__init__()
        self.regression = nn.Linear(n_feature, 1)
        
        
    def forward(self, x):
        linear_pred = self.regression(x)
        return torch.flatten(torch.sigmoid(linear_pred))
        

model = LogisticRegression(3)
x = torch.tensor([0, 0, 0], dtype=torch.float)
y_pred = model.forward(x)
```

6. **Expliquer comment pytorch sait quelles valeurs sont des paramètres et quelles valeurs ne sont pas des paramètres quand on crée un model avec une classe : en gros, comment fait pytorch pour distinguer les informations qui sont des paramètres (avec le gradient) et ceux dont il ne faut pas update le gradient ?**

Quand on set requires_grad à True dans un Tensor, Pytorch commence à construire un graphe qui va suivre chaque opération appliquée sur chaque noeud pour calculer le gradient. C'est possible grâce à la classe Autograd de Pytorch qui permet de calculer les dérivées et de constuire en mémoire le graphe

(https://towardsdatascience.com/pytorch-autograd-understanding-the-heart-of-pytorchs-magic-2686cd94ec95#:~:text=Gradients%20are%20calculated%20by%20tracing,way%20using%20the%20chain%20rule%20.)

7. **Dans cet exercice on va recoder l'optimiser SGD de pytorch pour comprendre comment ça marche. Coder la class SGD qui héritera de torch.optim.Optimizer. Elle devra avoir les méthodes suivantes :**
 **- un constructeur qui prendra les paramètres d'un modèle et l'argument lr (pas de momentum)**

 **- une méthode step qui met à jour les gradients**

 **- zero_grad**

```python
import torch


class SGD(torch.optim.Optimizer):
    
    def __init__(self, model_params, lr):
        super(SGD, self).__init__(model_params, {})
        self.params = model_params
        self.lr = lr
        
        
    def step(self):
        for param in self.params:
            param = param - self.lr * param.grad.data
            
        
    def zero_grad(self):
        for param in self.params:
            param.grad.data = torch.zeros_like(param.grad)
```

8. 