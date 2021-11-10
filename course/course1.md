# Deep Learning - Course 1

## Introduction

- **Detection** : isoler des objets sans les contours (juste les carrés autour)

- **Segmentation** : detection avec les contours

- **Donnees tabulaires possibles** : plutôt boosting, xgboost, ensembling

- **Graphes** : ex relations sur les reseaux sociaux

- **Auto-encodeur** : prend l'image en input et sort en output la reconstruction de l'image le mieux possible. 2 parties : encodeur qui génère une donnée qui aura une plus petite dimension que la donnée initiale, l'encodeur sert de moteur de compression de données, ensuite le décodeur doit prendre l'image compressée et doit sortir l'image originale la mieux reconstruite. Réduction de dimension possible avec les auto-encodeurs : possible pour les vidéos au lieu d'utiliser JPEG pour les images par exemple

- **Denoising auto-encoder** : on prend une image bruitée et on la met dans l'auto-encodeur et on veut reconstruire l'image débruitée. Il faut les 2 versions de l'image (bruitée et non bruitée) => plutôt supervisé.

- **GAN (generative adversarial networks)** : 1 RN (générateur) et 1 RN (discriminateur). Le générateur fournit des images et le discriminateur doit déterminer si c'est des vraies images ou non, le but du générateur est de générer des images de plus en plus difficiles à distinguer pour le discriminateur (dataset de vraies images et de fausses images ) => supervisé

- **Machine learning multi-modal** : en input, données sous plusieurs modalités (image, texte, son, etc.) puis classification d'image par ex ou donner image

- **Multi-view machine learning** : 

- **Deep learning et multi-task ML** : modèle entraîné à faire plsrs choses, ex classification d'image, traduction auto, etc.

- **Reinforcement learning** : entraîner un agent à avoir un comportement attendu

## Course

### Qu'est-ce que le deep learning ?

**Réseau de neurones** : profond cad avec de nbreuses couches

**Neurone / perceptron** : dérivé = neurone sigmoid (fonction mathématique) qui est composé d'une partie linéaire et d'une partie non linéaire. Prédiction linéaire avec vecteur X = [X1, X2, X3, ..., Xn]. La partie linéaire fait une combi linéaire des inputs avec des w1, w2, w3, ..., wn => w0 + w1 . X1 + ... + wn . Xn puis fonction non linéaire qui fait l'output du neurone. Fonctions non linéaires : escalier, sigmoid, ReLu. Sigmoid entre 0 et 1 donc utile dans classification linéaire. Dans la fonction de la régression logistique c'est la proba d'une classe qui sort. ReLu = max (0, 1). Dans un neurone il y a tjrs une partie linéaire et une partie non linéaire.

- **Ordonner neurones en couches** : 
  - Fully connected networks : chaque neurone d'une couche sera connecté à ts les neurones de la couche suivante. Le nb de couches d'un réseau est déterminé par l'user. Pas de connexion entre couches non successives.

- **Ensembling** :

- **Stacking** : un neurone = 1 petit modèle avec ses propres params et un réseau de neurones est un ens de neurones ordonnés en couches

- **TensorFlow Keras** : sous-module de TensorFlow pour faire des RN. Pour spécifier un modèle il faut nombre de couches avec nb de neurones par couches mais aussi fonction non linéaire (ex : ReLu) qui sera appliquée à tous les neurones de la même couche => fonction d'activation. Il faut aussi spécifier la méthode d'optimisation (ex : stochastic gradient) et la loss (prediction error of Neural Net => loss function)

- **Image en input** : voir enregistrement























