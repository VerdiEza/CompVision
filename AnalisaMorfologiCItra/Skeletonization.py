!pip install scikit-image
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, morphology

# Baca citra dari URL atau upload dari komputer
img_url = "/content/Foto Verdi.JPG"  # Ganti dengan URL citra Anda atau path lokal
image = io.imread(img_url)

# Ubah citra ke grayscale jika tidak sudah dalam grayscale
gray_image = color.rgb2gray(image)

# Binarisasi citra (mengubah ke citra hitam-putih)
binary_image = gray_image > 0.5  # Anda bisa mengubah nilai ambang sesuai kebutuhan

# Lakukan skeletonization
skeleton = morphology.skeletonize(binary_image)

# Tampilkan citra asli, citra biner, dan hasil skeletonization
plt.figure(figsize=(12, 4))
plt.subplot(131), plt.imshow(gray_image, cmap='gray'), plt.title('Citra Asli (Grayscale)')
plt.subplot(132), plt.imshow(binary_image, cmap='gray'), plt.title('Citra Biner')
plt.subplot(133), plt.imshow(skeleton, cmap='gray'), plt.title('Hasil Skeletonization')
plt.show()
