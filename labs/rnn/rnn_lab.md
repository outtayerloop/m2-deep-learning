# Deep learning avec Keras III : Réseaux de neurones récurrents


1. **Créer une classe RecurrentNeuron qui permet de simuler un neurone récurrent (unique). Le neurone ingérera une séquence représenté par un vecteur élément par élément grâce à la fonction predict. Elle renverra également un unique nombre. La sortie du neurone sera calculée avec la formule suivante :** 
&nbsp;

    **h_new = tanh(wh * h + wx * x)**
    &nbsp;

    **^y = wyh * h_new**

**La classe aura : une méthode __init__(self) permettant d'initialiser w_hh, w_x et w_yh aléatoirement. le hidden state sera initialisé à 0. une méthode set_w_h, set_w_x, set_w_yh permettant de spécifier paramètres (scalaire) manuellement une méthode d'instance predict(x) permettant de calculer la sortie d'un neurone. une méthode reset_hiden_state permettant de réinitialiser le hiden state à 0. une méthode get_hiden_state permettant de récupérer l'hiden state h. Instancier votre classe dans une variable neuron et appeler plusieurs fois la fonction predict(0.1) afficher la variable h après chaque prédicte et vérifier qu'elle change bien entre deux prédictions**

```python
import numpy as np
import random


class RecurrentNeuron:
    
    def __init__(self):
        self.wh = random.uniform(0,1)
        self.wx = random.uniform(0,1)
        self.wyh = random.uniform(0,1)
        self.h = 0
        
        
    def set_w_h(self, new_wh):
        self.wh = new_wh
        
    
    def set_w_x(self, new_wx):
        self.wx = new_wx
        
        
    def set_w_yh(self, new_wyh):
        self.wyh = new_wyh
    
    
    def predict(self, x):
        h_new = np.tanh(self.wh * self.h + self.wx * x)
        self.h = h_new
        return self.wyh * self.h
        
        
    def get_w_h(self):
        return self.wh
        
    
    def get_w_x(self):
        return self.wx
        
        
    def get_w_yh(self):
        return self.wyh
    
        
    def get_hidden_state(self):
        return self.h
        
        
neuron = RecurrentNeuron()
neuron.predict(0.1)
print(neuron.get_hidden_state())
neuron.predict(0.1)
print(neuron.get_hidden_state())
neuron.predict(0.1)
print(neuron.get_hidden_state())
```

2. **Explain how a Recurent neurone works and the greatest difference with a Feedforward neuron**

At a given time, a recurrent neuron uses both its input and the state of the previous neuron to predict its output, it allows parameter sharing and  introduces the notion of sequential memory (similar to short-term memory). Its training is similar to a simple feed forward neuron except the backdrop is applied for each data point in the given sequence. It is sometimes called the BTT (Back propagation Through Time).

3. **LSTM and GRU are extensions classical RNN neurones.  Explain what is the main problem of standard RNN neuron and how LSTM solve this problem. Explain the difference between GRU / LSTM cell and classical (also named vanilla) RNN.**

- The main problem of a standard RNN neuron is short-term memory caused by the vanishing gradient problem (VGP), as the RNN processes more data, it has more trouble retaining information from previous steps. It comes from the nature of back propagation which adjusts the weights by calculating the gradient of each node in the NN, the bigger the gradient, the bigger the adjustments are and vice-versa. The problem is, in RNNs, back propagation works by calculating the gradient of a node from the gradient of the previous one, which means if the previous gradient was small, then it would cause the next gradients to be shrinked exponentially, leading to the vanishing gradient problem with no adjustment made to the RNN and thus short-term memory. LSTMs are able to learn long-term dependencies using a mechanism called gates which are different tensor operations that learn information that can learn what information to add or remove to the hidden state of the feedback loop.

- Main difference between GRNN and LSTM : GRNN has 2 gates to control its memory (update gate and reset gate) whereas LSTMs have 3 gates (update gate, reset gate and forget gate). The update gate is the input gate, the reset is the output gate.


4. **Créer une couche Recurrente SimpleRNN avec Keras ayant 30 neurones.**

```python
from tensorflow.keras.layers import SimpleRNN



layer = SimpleRNN(30)
```

5. **De nos jours les gens n'utilisent plus les couches RNN basiques car elles ont des défauts (elles ne marchent pas bien sur les séquences longues). Créer une couche LSTM avec 30 neurones.**

```python
from tensorflow.keras.layers import LSTM



layer = LSTM(30)
```

6. **Les réseaux de neurones LSTM sont constitués de deux parties : les couches récurrents et les couches Dense.  Créer une variable model contenant un réseau de neurones avec : Une couche LSTM avec 10 neurones, Une couche Flatten, Une couche dense avec 1 neurone. Compiler le modèle avec l'optimiser sgd et la loss adéquate pour faire de la régression**

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten


model = Sequential()
model.add(LSTM(10))
model.add(Flatten())
model.add(Dense(1))


model.compile(optimizer='sgd', loss='mse')
```

7. **Les réseaux de neurones récurrent s'attendent à avoir des données séquentielles en entrée. Une couche LSTM s'attend à avoir une séquence de vecteur en entrée. Les features doivent être des données sous la forme (nombre_exmple, longueur_sequence, dimension_de_chaque_vecteur). On dispose d'une fonction load_data qui renvoie un tuple avec un X et un y. Créer une une variable X_train et y_train en appelant la fonction load_data. Afficher la taille de X_train. Est-elle conforme à ce que le réseau s'attend ? Mettre à jour X_train avec la méthode reshape pour que les données soient dans le bon format.**

```python
X_train, y_train = load_data()
print(X_train)
print(X_train.shape)
seq_len = X_train.shape[0]
vect_dim = X_train.shape[1]

X_train = X_train.reshape(seq_len, vect_dim, -1)
```

Les LSTMs attendent une shape 3D, or la shape initiale de X_train était (5,4) donc on doit reshape en laissant la 3eme dimension être dynamiquement déterminée (d'où le -1 en 3eme position du reshape).

8. Temporary (does not pass test) :

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten


X_train, y_train = load_data()
seq_len = X_train.shape[0]
vect_dim = X_train.shape[1]
X_train = X_train.reshape(seq_len, vect_dim, -1)

model = Sequential()
model.add(LSTM(units=10))
model.add(Flatten())
model.add(Dense(units=1, activation='linear'))

model.compile(optimizer='sgd', loss='mse')

model.fit(X_train, y_train, epochs=2)
```

