import cv2
import numpy as np

# Load the image
image_path = '/content/Foto Verdi.JPG'
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform corner detection using the Harris corner detection algorithm
corner_detection = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)
corner_detection = cv2.dilate(corner_detection, None)

# Define a threshold for identifying corners
threshold = 0.01 * corner_detection.max()

# Mark the detected corners with a red color
marked_image = image.copy()
marked_image[corner_detection > threshold] = [0, 0, 255]  # Red color for corners

# Display the original and marked images using Matplotlib
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
plt.title('Corner Detection')
plt.axis('off')

plt.tight_layout()
plt.show()
