- **Convoluted image size** : 
```
conv_size = (W – F + 2P) / S + 1
```
With :
```
W : width of the original image, H for height
F : kernel size (height or width depending on current W or H)
P : padding
S : stride
```

- Stride : translation sur l'axe des x ET sur l'axe des y sur l'image originelle