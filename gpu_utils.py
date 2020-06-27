import torch
from image_processor import *
dct_matrix = dct_matrix_creator()
img = cv2.imread("test.png")
height = img.shape[0]
width = img.shape[1]
b, g, r = cv2.split(img)