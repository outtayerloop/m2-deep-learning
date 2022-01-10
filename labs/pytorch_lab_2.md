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

(Test n° 1 -- 'to be filled')

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

x = torch.tensor([1, 1], dtype=torch.float)
y_pred = model.forward(x)

parameters = model.parameters()
for param in parameters:
    print(param)
```

4. 