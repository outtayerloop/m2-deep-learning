# CM 2 THeory Lab

1. Explain what is gradient descent

It is an optimization algorithm that aims to minimize the loss function by iteratively changing the bias and the weights of each variable (the coefficient matrix) to decrease the loss function slope

2. Explain what is back-propagation and its use for the training of neural networks

Back propagation uses gradient descent after having computed the loss from the model's fit, and it goes from the last layer to the first one, therefore it is ony used in neural networks. However, gradient descent can be adapted to most models generally

3. **Explain what is automatic différentiation and how it works. Which famous deep learning tools use automatic differentiation ?**

TensorFlow, PyTorch
=> ne donne pas la formule du gradient, sert à calculer une valeur numérique du gradient automatiquement. Utilisé par le fit pour calculer les gradients/dérivées partielles, utilisé pour faire de la descente de gradient