import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Baca citra
img = cv2.imread('/content/Foto Verdi.JPG', 0)  # Ganti 'input_image.jpg' dengan nama citra Anda
cv2_imshow( img)

# Membuat kernel untuk dilasi
kernel = np.ones((5, 5), np.uint8)  # Ubah ukuran kernel sesuai kebutuhan

# Melakukan dilasi
dilation = cv2.dilate(img, kernel, iterations=1)

# Menampilkan citra hasil dilasi
cv2_imshow(dilation)

cv2.waitKey(0)
cv2.destroyAllWindows()
