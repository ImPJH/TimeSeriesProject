import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pywt
import pywt.data

img_gray = cv2.imread('source/4.2.07.tiff', cv2.IMREAD_GRAYSCALE)
print(img_gray)
print('img shape = ',img_gray.shape)
plt.imshow(img_gray,cmap='gray')
plt.show(block=False)

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(img_gray, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()

# Read residual_2.csv
df2 = pd.read_csv('source/residual_2.csv', header=None)
residual = df2.values
print('================\nresidual', residual)
print('residual shape', residual.shape)