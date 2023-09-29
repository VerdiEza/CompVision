import cv2
import numpy as np
from google.colab import files
import matplotlib.pyplot as plt

# Upload an image file from your local machine to Colab
uploaded = files.upload()

# Read the uploaded image
for filename in uploaded.keys():
    image_path = filename

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply thresholding using cv2.threshold()
_, thresholded_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Display the original and thresholded images using Matplotlib
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(thresholded_image, cmap='gray')
plt.title('Thresholded Image')
plt.axis('off')

plt.tight_layout()
plt.show()
