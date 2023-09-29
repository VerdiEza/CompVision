import cv2
import numpy as np

# Baca citra biner
binary_image = cv2.imread('/content/Foto Verdi.JPG', cv2.IMREAD_GRAYSCALE)

# Labeling wilayah-wilayah
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

# Loop melalui wilayah-wilayah yang telah diberi label
for label in range(1, num_labels):
    area = stats[label, cv2.CC_STAT_AREA]
    perimeter = cv2.arcLength(np.array(np.where(labels == label)).T, True)

    print(f"Wilayah {label}: Luas = {area}, Keliling = {perimeter}")
