import cv2
import numpy as np
from matplotlib import pyplot as plt

# Baca citra dari Google Colab
img_path = "/content/Foto Verdi.JPG"  # Ganti dengan path citra Anda di Google Colab
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Membuat kernel untuk opening
kernel = np.ones((5, 5), np.uint8)  # Ubah ukuran kernel sesuai kebutuhan

# Melakukan opening
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# Menampilkan citra asli dan citra hasil opening
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Citra Asli')
plt.subplot(122), plt.imshow(opening, cmap='gray'), plt.title('Hasil Opening')
plt.show()
