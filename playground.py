from image_processor import *
import numpy as np
from scipy.fft import dct, idct
import os
from matplotlib import pyplot as plt
import cv2


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
    img = cv2.imread("test.png")
    print(img.shape)
    height = img.shape[0]
    width = img.shape[1]
    b, g, r = cv2.split(img)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y_channel, u_channel, v_channel = cv2.split(img_yuv)
    rectified_b = channel_regulator(b, height, width)
    rectified_y_channel = channel_regulator(y_channel, height, width)
    rectified_u_channel = channel_regulator(u_channel, height, width)
    rectified_v_channel = channel_regulator(v_channel, height, width)
    print("Original Y channel:")
    print(rectified_y_channel)
    rectified_b_copy = rectified_b
    rectified_g = channel_regulator(g, height, width)
    rectified_r = channel_regulator(r, height, width)
    cv2.imwrite("org.png", rectified_y_channel)
    height = rectified_b.shape[0]
    width = rectified_b.shape[1]
    shape = (height // 8, width // 8, 8, 8)
    strides = rectified_y_channel.itemsize * np.array([width * 8, 8, width, 1])
    blocks = np.lib.stride_tricks.as_strided(rectified_y_channel, shape=shape, strides=strides)
    quantization_matrix = np.load(os.path.join("quantization_tables", "Adobe_Photoshop__Save_As_11_lc.npy"))
    print("Quantization Matrix")
    print(quantization_matrix.reshape(8,8))
    for i in range(height // 8):
        for j in range(width // 8):
            if i == 23 and j == 23:
                print("Original Block")
                print(blocks[i, j])
                print("Shifted Block")
                print(blocks[i, j]-127)
                print("DCT Block")
                dctd = dct(blocks[i, j]-127)
                print(dctd)
                print("IDCT Block")
                print(idct(dctd))
                print("Reversed Shfting Block")
                print(idct(dctd)+127)
            blocks[i, j] = quantizer(dct2(blocks[i, j] - 127), quantization_matrix)
            if i == 23 and j == 23:
                print("DCT and Quantilized Matrix")
                print(blocks[i, j])
            blocks[i, j] = blocks[i, j] * quantization_matrix.reshape(8, 8)
            if i == 23 and j == 23:
                print("Reverse Quantilized Matrix")
                print(blocks[i, j])
                print("IDCT Matrix")
                print(idct2(blocks[i,j]).astype(int))
            blocks[i,j] = idct2(blocks[i,j]).astype(int)+127
            if i == 23 and j == 23:
                print("IDCT and Reshifted Matrix")
                print(blocks[i, j])
            pass
    res = np.hstack(np.hstack(blocks))
    print("Recovered Block")
    print(blocks[23][23])
    # res = idct(res)
    # res = res+127
    img = cv2.merge([res, rectified_u_channel, rectified_v_channel])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
    print("Recovered Y channel")
    print(res)
    cv2.imwrite("recovered.png", res)
    plt.subplot(1, 2, 1), plt.imshow(res, cmap='gray')
    plt.title('input image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(img_rgb)
    plt.title('magnitude spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
