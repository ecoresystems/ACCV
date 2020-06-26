import os
import time

from matplotlib import pyplot as plt
from numpy import r_

from image_processor import *

if __name__ == "__main__":
    reference_quanti_matrix = np.array(
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
    rectified_y_channel = channel_regulator(y_channel, height, width)
    rectified_u_channel = channel_regulator(u_channel, height, width)
    rectified_v_channel = channel_regulator(v_channel, height, width)
    imsize = rectified_y_channel.shape
    idct = np.zeros(imsize)
    rectified_y_channel_shifted = rectified_y_channel.astype(dtype="int16") - 128
    quantization_matrix = np.load(os.path.join("quantization_tables", "Adobe_Photoshop__Save_As_00_lc.npy")).reshape(8,                                                                                                      8)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            dct_block = dct_transformer(rectified_y_channel_shifted[i:(i + 8), j:(j + 8)], dct_matrix)
            quantified_block = np.round(dct_block / quantization_matrix)
            restored_dct_block = quantified_block * quantization_matrix
            restored_block = idct2(restored_dct_block)
            idct[i:(i + 8), j:(j + 8)] = block_processor().astype(dtype="uint8")
    reshifted_y_channel = (idct + 127).astype(dtype="uint8")
    print("Final y channel:")
    print(reshifted_y_channel)
    concatenate_res = np.hstack((reshifted_y_channel, rectified_y_channel))
    plt.imshow(concatenate_res, cmap="gray")
    cv2.imwrite("PROCESSED_REC_Y.png", reshifted_y_channel)
    restored_img = cv2.merge([reshifted_y_channel, rectified_u_channel, rectified_v_channel])
    img_rgb = cv2.cvtColor(restored_img, cv2.COLOR_YUV2RGB)
    # cv2.imwrite("recovered.png", reshifted_y_channel)
    plt.subplot(1, 2, 1), plt.imshow(img_rgb)
    plt.title('input image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('magnitude spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
