import os

from matplotlib import pyplot as plt

from image_processor import *

if __name__ == "__main__":
    # This is the quantization matrix provided by Joint Expert Group, for general reference purpose
    reference_quantization_matrix = np.array(
        [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 59, 14, 17, 22, 29,
         51,
         87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120,
         101,
         72, 92, 95, 98, 112, 100, 103, 99])

    img = cv2.imread("test.png")
    img = cv2.resize(img, None, fx=0.6, fy=0.6)
    dct_matrix = dct_matrix_creator()
    quantization_matrix_lc = np.load(os.path.join("quantization_tables", "Adobe_Photoshop__Save_As_00_lc.npy")).reshape(
        8, 8)
    quantization_matrix_cc = np.load(os.path.join("quantization_tables", "Adobe_Photoshop__Save_As_00_lc.npy")).reshape(
        8, 8)
    processed_img = background_processor(img, dct_matrix, quantization_matrix_lc, quantization_matrix_lc)
    print(processed_img.max())
    print(processed_img.min())
    img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_YUV2RGB)
    plt.subplot(1, 2, 1), plt.imshow(img_rgb)
    plt.title('Processed Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.show()
