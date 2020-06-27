from image_processor import *
import numpy as np
from scipy.fft import dct, idct
import time
import os
import torch
from matplotlib import pyplot as plt
import cv2


def dct2(a):
    return dct(dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')
print(torch.cuda.current_device())
test_matrix = np.full((1,64),222)
print(test_matrix)
block = np.array(
    [52, 55, 61, 66, 70, 61, 64, 73, 63, 59, 55, 90, 109, 85, 69, 72, 62, 59, 68, 113, 144, 104, 66, 73, 63, 58, 71,
     122, 154, 106, 70, 69, 67, 61, 68, 104, 126, 88, 68, 70, 79, 65, 60, 70, 77, 68, 58, 75, 85, 71, 64, 59, 55, 61,
     65, 83, 87, 79, 69, 68, 65, 76, 78, 94])

quantization_matrix = np.array(
    [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 59, 14, 17, 22, 29, 51,
     87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101,
     72, 92, 95, 98, 112, 100, 103, 99])
shifted = test_matrix.reshape(8, 8) - 128
print(shifted)
dctd = dct2(shifted)
dct_matrix = dct_matrix_creator()
dct_block_by_calculation = dct_transformer(shifted, dct_matrix)
print(dct_block_by_calculation)
quantified_res = quantizer(dct_block_by_calculation, quantization_matrix)
print(quantified_res)
# ------------------- Here, we start to reverse this process ---------------------------#
restored_dct_matrix = quantified_res * quantization_matrix.reshape(8, 8)
print(restored_dct_matrix)
reversed_matrix = idct_transformer(restored_dct_matrix, dct_matrix)
print(reversed_matrix.astype(int))
print(reversed_matrix.astype(int) + 128)
final_img = np.hstack((block.reshape(8,8), reversed_matrix.astype(int) + 128))
plt.imshow(final_img, cmap="gray")
plt.show()
