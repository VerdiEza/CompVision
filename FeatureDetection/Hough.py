import cv2
import numpy as np
from google.colab import files
from matplotlib import pyplot as plt

# Upload an image file from your local machine to Colab
uploaded = files.upload()

# Read the uploaded image
for filename in uploaded.keys():
    image_path = filename

image = cv2.imread(image_path, cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise and improve Hough transform accuracy
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Perform standard Hough Line Transform
edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

# Draw detected lines on a copy of the original image
image_with_lines = image.copy()
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Perform Hough Circle Transform
circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                           param1=50, param2=30, minRadius=10, maxRadius=50)

# Draw detected circles on a copy of the original image
image_with_circles = image.copy()
if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        center = (circle[0], circle[1])
        radius = circle[2]
        cv2.circle(image_with_circles, center, radius, (0, 255, 0), 2)

# Display the original image, lines, and circles using Matplotlib
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB))
plt.title('Hough Lines')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(image_with_circles, cv2.COLOR_BGR2RGB))
plt.title('Hough Circles')
plt.axis('off')

plt.tight_layout()
plt.show()
