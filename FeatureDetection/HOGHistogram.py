import cv2
import numpy as np
from google.colab import files
import matplotlib.pyplot as plt

# Upload an image file from your local machine to Colab
uploaded = files.upload()

# Read the uploaded image
for filename in uploaded.keys():
    image_path = filename

image = cv2.imread(image_path, cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a HOG descriptor object
hog = cv2.HOGDescriptor()

# Compute HOG features
hog_features = hog.compute(gray_image)

# Plot the HOG features as a histogram
plt.figure(figsize=(8, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.hist(hog_features, bins=36)
plt.title('HOG Features Histogram')
plt.xlabel('Bins')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
