from image_processor import *
import numpy as np

block = np.random.random((8, 8))
block = np.floor(block*255).astype(int)
dct_matrix = dct_matrix_creator()

res = np.dot(dct_matrix, block)
res = np.dot(res, np.transpose(dct_matrix))

print(res == dct_matrix*block*dct_matrix.T)
print(res)
print(dct_matrix*block*(dct_matrix.T))