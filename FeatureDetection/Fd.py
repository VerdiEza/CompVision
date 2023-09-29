#import library
import cv2
import matplotlib.pyplot as plt

im = cv2.imread('/content/Foto Verdi.JPG') #Sesuaikan dengan nama file
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
plt.imshow(im, cmap='gray')
plt.show()

#Menerapkan filter canny
filter_canny = cv2.Canny(im,25,255,L2gradient=False)
#Menerapkan filter sobel
filter_sobel = cv2.Sobel(src=im, ddepth=cv2.CV_8U, dx=1, dy=1, ksize=5)

plt.imshow(filter_canny, cmap='gray')
plt.show()

plt.imshow(filter_sobel, cmap='gray')
plt.show()
