# Deep Learning - Lab 1

1. **Expliquer la différence entre Le machine learning Multi-task, multi-modal et multi-view**

- Machine learning multi-task : modèle entraîné à faire plsrs choses, ex classification d'image, traduction auto, etc. Faire plsrs tâches différentes pr un seul et même modèle.

- Machine learning multi-modal : en input, données sous plusieurs modalités (image, texte, son, etc.) puis classification d'image par ex ou donner image, ex : image, son etc en input et output sous forme de video par ex. Le machine Learning multi modal est le fait de pouvoir apprendre automatiquement de plusieurs canaux comme le son, le texte et les images. Il faut cependant faire attention à l'importance donnée à chaque réseau. L'apprentissage mullti-view est un apprentissage à l'aide d'image qui sont tournées dans n'importe quels sens afin d'être capable de reconnaitre sous toutes les visons une images. L'objectif du machin learning multi-task est de combiner les tâches simultanément pour améliorer la performance de prédiction par rapport à l’apprentissage de ces tâches de manière indépendante.

- Machine learning multi-view : plusieurs datasets en entrée. Par ex ML sur une image pour reconnaître un bus sous plusieurs angles ou plsrs angles d'une maison (montage 3D). Le modèle a plusieurs perspectives d'une même image

2. **Expliquer la particularité du machine learning sur des données de type graphes**

Quels outils existent pour du Deep learning sur les graphes ? 

Donner un ou deux cas d'application intéressants

   - ML sur graphes : extraction de features manuelle (par l'utilisateur), ex : recommandation de relations sur les réseaux sociaux, design de protéines / molécules / médicaments

   - Deep Learning sur graphes : extraction features automatique. Outils : GeometricFlux.jl, PyTorch GNN, Jraph, Spektral, Graph Nets, Deep Graph Library (DGL) et PyTorch Geometric. Ex : vision par ordinateur, classification d'ECG

3. **Expliquer le principe d'un auto-encodeur et donner des exemples concrets d'applications de cette méthode** 

Auto-encodeur : prend l'image en input et sort en output la reconstruction de l'image le mieux possible. 2 parties : encodeur qui génère une donnée qui aura une plus petite dimension que la donnée initiale, l'encodeur sert de moteur de compression de données, ensuite le décodeur doit prendre l'image compressée et doit sortir l'image originale la mieux reconstruite. Réduction de dimension possible avec les auto-encodeurs : possible pour les vidéos au lieu d'utiliser JPEG pour les images par exemple.

4. **Rappeler ce qu'est la régularisation L1 / L2. Expliquer quels sont les mécanismes de régularisation spécifiques aux réseaux de neurones**

La régularisation L1 / L2 permet d'éviter l'overfitting. Elle diminue la complexité d'un réseau de neurones durant l'entraînement. La régularisation L2 permet de réarranger les poids à chaque update du réseau de neurones, la L1 permet de faire de même mais avec une méthode mathématique différente.

5. **En faisant le lien avec les architecture des GPU, expliquer pourquoi l'entraînement et la prédiction des modèles de deep learning est significativement plus rapide sur GPU que sur CPU**

Les GPUs sont optimisés pour le deep learning parce qu'ils permettent d'effectuer plusieurs calculs simultanément (ils ont plus de coeurs) donc permettent le traitement parallélisé et donc d'aller plus vite au niveau de l'entraînement des réseaux de neurones en deep learning.

6. **Expliquer ce qu'est la rétropagation du gradient**

La rétro-propagation consiste en mettre à jour les poids de chaque noeud du réseau de neurones après un test sur un input donné pour permettre la correction des erreurs, en partant de la dernière couche vers la 1ère et en corrigeant successivement les poids en fonction de la décision prise par chaque couche par rapport à l'output attendu.

7. **Expliquer a quoi sert la différentiation automatique et comment elle marche de manière générale**

Utile pour créer et entraîner des modèles de deep learning complexes sans avoir à optimiser manuellement les modèles, notamment pour leur customization, faire des boucles d'entraînement et des fonctions de loss. Contrairement à la méthode manuelle, la différentiation automatique ne construit pas d'expression symbolique pour les dérivées

