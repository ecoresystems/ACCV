from image_processor import *
import numpy as np
from scipy.fft import dct, idct
import os
from numpy import r_
from matplotlib import pyplot as plt
import cv2
import time


def do_sth():
    block = np.random.random((8, 8))
    block = np.floor(block * 255).astype(int)
    dct_matrix = dct_matrix_creator()

    quantization_matrix = np.load(os.path.join("quantization_tables", "Adobe_Photoshop__Save_As_00_lc.npy"))

    print("Original Value:")
    print(block)

    print("DCT VALUE:")
    dct_res = dct(block - 128)
    print(dct_res)
    print("Quantization Matrix:")
    print(quantization_matrix.reshape(8, 8))
    print("Quantization Result:")
    quan_res = np.rint(dct_res / quantization_matrix.reshape(8, 8))
    print(quan_res)
    print("Restored Res:")
    print(idct(quan_res).astype(int) + 128)
    # print(idct(np.round(dct(block - 128)/quantization_matrix.reshape(8,8))).astype(int) + 128)


if __name__ == "__main__":
    reference_quanti_matrix = quantization_matrix = np.array(
        [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 59, 14, 17, 22, 29,
         51,
         87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120,
         101,
         72, 92, 95, 98, 112, 100, 103, 99])
    dct_matrix = dct_matrix_creator()
    img = cv2.imread("test.png")
    height = img.shape[0]
    width = img.shape[1]
    b, g, r = cv2.split(img)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y_channel, u_channel, v_channel = cv2.split(img_yuv)
    rectified_b = channel_regulator(b, height, width)
    rectified_y_channel = channel_regulator(y_channel, height, width)
    rectified_u_channel = channel_regulator(u_channel, height, width)
    rectified_v_channel = channel_regulator(v_channel, height, width)
    print(rectified_y_channel.shape)
    print(rectified_u_channel.shape)
    print(rectified_v_channel.shape)
    cv2.imwrite("Original_REC_Y.png",rectified_y_channel)
    rectified_b_copy = rectified_b
    rectified_g = channel_regulator(g, height, width)
    rectified_r = channel_regulator(r, height, width)
    cv2.imwrite("org.png", rectified_y_channel)
    height = rectified_b.shape[0]
    width = rectified_b.shape[1]
    imsize = rectified_y_channel.shape
    dct = np.zeros(imsize)
    idct = np.zeros(imsize)
    rectified_y_channel_shifted = rectified_y_channel.astype(int) -128
    print("Shifted y channel")
    print(rectified_y_channel_shifted)
    quantization_matrix = np.load(os.path.join("quantization_tables","Adobe_Photoshop__Save_As_00_lc.npy")).reshape(8, 8)
    start_time = time.time()
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            dct_block = dct2(rectified_y_channel_shifted[i:(i + 8), j:(j + 8)])
            quantified_block = np.round(dct_block / quantization_matrix)
            restored_dct_block = quantified_block * quantization_matrix
            restored_block = idct2(restored_dct_block)
            idct[i:(i + 8), j:(j + 8)] = restored_block.astype(dtype="uint8")
    reshifted_y_channel = (idct + 127).astype(dtype="uint8")
    print("Time consumption per channel: ",end="")
    print(time.time()-start_time)
    print("Final y channel:")
    print(reshifted_y_channel)
    concatenate_res = np.hstack((reshifted_y_channel, rectified_y_channel))
    plt.imshow(concatenate_res, cmap="gray")
    cv2.imwrite("PROCESSED_REC_Y.png",reshifted_y_channel)
    restored_img = cv2.merge([reshifted_y_channel, rectified_u_channel, rectified_v_channel])
    img_rgb = cv2.cvtColor(restored_img, cv2.COLOR_YUV2RGB)
    # cv2.imwrite("recovered.png", reshifted_y_channel)
    plt.subplot(1, 2, 1), plt.imshow(img_rgb)
    plt.title('input image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('magnitude spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
