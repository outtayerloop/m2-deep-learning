# Convolution walkthrough lab

1. **Le fichier image.jpg disponible dans votre environnement Vous avez également à votre disposition la variable conv_filter représentant un noyau de convolution (tableau numpy). Appliquez une convolution sur image à l'aide de la fonction filter2D d'OpenCV et stockez le résultat dans la variable convoluted_image. Vous veillerez à appliquer les paramètres suivants: ddepth = -1 ; kernel = conv_filter**

```python
import numpy as np
import cv2
conv_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
img = cv2.imread('image.jpg')
convoluted_image = cv2.filter2D(img, -1, conv_filter)
```

2. 