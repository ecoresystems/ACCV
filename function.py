import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('test.jpg',0)
print(img)
# img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
plt.subplot(1, 2, 1), plt.imshow(img)
plt.title('input image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(magnitude_spectrum)
plt.title('magnitude spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
