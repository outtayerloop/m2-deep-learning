# CM 2 THeory Lab

1. **Explain what is gradient descent**

It is an optimization algorithm that aims to minimize the loss function by iteratively changing the bias and the weights of each variable (the coefficient matrix) to decrease the loss function slope

2. **Explain what is back-propagation and its use for the training of neural networks**

Back propagation uses gradient descent after having computed the loss from the model's fit, and it goes from the last layer to the first one, therefore it is ony used in neural networks. However, gradient descent can be adapted to most models generally

3. **Explain what is automatic différentiation and how it works. Which famous deep learning tools use automatic differentiation ?**

TensorFlow, PyTorch
=> ne donne pas la formule du gradient, sert à calculer une valeur numérique du gradient automatiquement. Utilisé par le fit pour calculer les gradients/dérivées partielles, utilisé pour faire de la descente de gradient. Utile pour créer et entraîner des modèles de deep learning complexes sans avoir à optimiser manuellement les modèles, notamment pour leur customization, faire des boucles d'entraînement et des fonctions de loss. Contrairement à la méthode manuelle, la différentiation automatique ne construit pas d'expression symbolique pour les dérivées. Pytorch fait partie des frameworks de différenciation automatique avec TensorFlow.

4. **Explain what is transfert learning, its main advantages and the various kind of transfert learning**

Transfer learning is a technique which aims to fasten model training. It consists of a pre-trained model from which we will make the dense layers untrainable for we will replace them by our current model's dense layers instead (fine-tuning). At the end, we'll have a model constituted by a pre-trained model's preprocessing layers and our model's classification or regression layers.

5. **Explain what is the functional API in keras and its advantages over the sequential API**

It allows computing the output of a layer and then plugging this output into the next one until we get to the end. It's about assigning a value issued from a function to a variable (for instance x) which will be reused again after and will be reassigned a new value from another function which will use the previous value of x. The functional API uses each layer as a function. Rather than building an object from which we cannot get each step's output like with the Sequential API, we can monitor each step with the functional API.