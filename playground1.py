from image_processor import *
import numpy as np
from scipy.fft import dct, idct
import os
from matplotlib import pyplot as plt
import cv2

block = np.array(
    [52, 55, 61, 66, 70, 61, 64, 73, 63, 59, 55, 90, 109, 85, 69, 72, 62, 59, 68, 113, 144, 104, 66, 73, 63, 58, 71,
     122, 154, 106, 70, 69, 67, 61, 68, 104, 126, 88, 68, 70, 79, 65, 60, 70, 77, 68, 58, 75, 85, 71, 64, 59, 55, 61,
     65, 83, 87, 79, 69, 68, 65, 76, 78, 94])

quantization_matrix = np.array(
    [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 59, 14, 17, 22, 29, 51,
     87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101,
     72, 92, 95, 98, 112, 100, 103, 99])
shifted = block.reshape(8,8)-128
print(shifted.reshape(64))
dctd = dct(shifted.reshape(64))
print(dctd)