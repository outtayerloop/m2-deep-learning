# OpenCV library review lab

1. **Opencv is a library coded in C++ which is very convienient for image manipulation. It has a python bindings so that you can use it in python too.**

   1. **Import the library (it is named cv2)**

   2. **With the imread, load the image image.jpg and store the result in the variable img.**

```python
import cv2

img = cv2.imread('image.jpg')
```

2. **Import cv2**

   1. **With the imread function, load the image image.jpg and store the result in the variable img.**

   2. **Update the value of the array : put 0 for every columns between 50 and 100 excluded**

```python
import cv2
import numpy as np

img = cv2.imread('image.jpg')
img[:, 50:100, :] = 0
```

3. **We have created for you a variable named img which contains an image. With opencv and the cvtColor function, convert the image into a black and white one. store the result in the variable bw_img**

```python
#img variable exists
import cv2

bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

4. **We want to resize an image stored in the variable img. With the resize function of open cv, resize the image so that it size is 200 by 200 pixels. Store it in a variable named img_resized. No need to use interpolation.**

```python
import cv2

img_resized = cv2.resize(img, (200,200))
```

5. **We have created an img variable containing an image.**

   1. **With the function getRotationMatrix, create a rotation matrix make a rotation of 60 degree. Store it in a variable named rotation_matrix The rotation image should be the center of the image.** 

   2. **Use the rotation image with warpAffine function. The target image should have the size as the original image. Store the result in a variable name rotated_image**

   3. **try with a rotation center of 0 once you have validated the exercice**

```python
import cv2

height, width = img.shape[:2]
print(height, width)
img_center = (height/2, width/2)
rotation_matrix = cv2.getRotationMatrix2D(center=img_center, angle=60, scale=1)
rotated_image = cv2.warpAffine(src=img, M=rotation_matrix, dsize=(width, height))

# Rotation center of 0
# Replace img_center by (0,0)
```

6. **We have created a variable bw_img containing a black and white image.**

   1. **Compute the histogram of the image using either numpy.histogram or matplotlib.hist and store the result in the variable hist.**
    

   2. **Display the the histogram using matplotlib.hist**

```python
import numpy as np
import matplotlib.pyplot as plt

print(bw_img.shape)
hist = np.histogram(bw_img)
plt.hist(bw_img)
plt.title('Image histogram')
plt.show()
```

7. **An image was loaded in the variable named img. Using the OpenCV's threshold method, tranform the image to black and white format. Take 127 as threshold. Store the result in a variable named thesholded. Warning: the threshold method of OpenCV returns two values. It is the second value that contains the thresholded image and it is this image that need to be stored in the wanted variable.**

```python
import cv2

thresholded = cv2.threshold(img,127,255, cv2.THRESH_BINARY)[1]
```

8. **Explain what is the Otsu method and how it improves upon basic thresholding techniques ?**

The Otsu method is used to do an automatic thresholding from the image histogram or to map an image to a grey (black/white) version to obtain a binary image. In thresholding we have to manually pass the threshold we want to use

9. **Filtering image. Sometimes you want to apply basic image filter to remove noise. Two well known filters are Gaussian filter and Median filter.**

   1.  **Load the image.jpg file with opencv and store the result in a variable img.**

   2.  **Apply a Gaussian filter on img  with the GaussianBlur function (use a kernel size of (5,5)). Store the result in a variable named gaussian_filtered**

   3.  **Apply a Median filter on img with the medianBlur function and store it in a variable named median_filtered**

   **Do you see a difference in images output ?**

```python
import cv2

img = cv2.imread('image.jpg')
gaussian_filtered = cv2.GaussianBlur(img,(5,5),0)
median_filtered = cv2.medianBlur(img, 5)

# The gaussian filtered image is more blurred
```

10. **The convolution operation is a very central operation for signal and image processing. Convolution can be extended to 2D or 3D signal like images and are used by convolutional neural networks. We loaded an image in a variable named image. The convolutionnal_kernel variable contains a small matrix representing a 2 dimension convolutional kernel.**  

    1. **Using OpenCV's filter2D function, compute the convolution of the image by the convolutional_kernel. Store the result in a variable named output**

    **You will ensure that the following settings are applied: ddepth:  -1;** 

    1. **Try to change the values of the kernel once you have validated the answer.** 

    2. **Can you find a matrix value that would result a blurry image like the gaussian blur ?**

```python
import numpy as np
import cv2

convolutional_kernel = np.array(
    [[-1, -1, -1], 
     [-1, 8, -1], 
     [-1, -1, -1]]
)

output = cv2.filter2D(image, -1, convolutional_kernel)

# Matrix value that would result in a blurry image like the
# gaussian blur
# src: https://fr.wikipedia.org/wiki/Noyau_(traitement_d%27image)
# It is the blurred gauss matrix 5 x 5
```

11. 


