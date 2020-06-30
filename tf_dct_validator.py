import tensorflow as tf
from scipy.fftpack import dct

from image_processor import *
from image_processor_gpu import image_regulator


def perform_blockwise_dct_tf(img):
    shape = tf.shape(img)
    x, y, c = shape[0], shape[1], shape[2]
    img_res = tf.reshape(img, [x // 8, 8, y // 8, 8, c])
    img_dct1 = tf.signal.dct(tf.transpose(img_res, [0, 1, 2, 4, 3]), norm='ortho')
    img_dct2 = tf.signal.dct(tf.transpose(img_dct1, [0, 2, 4, 3, 1]), norm='ortho')
    out = tf.reshape(tf.transpose(img_dct2, [0, 4, 1, 2, 3]), shape)
    return out


def perform_blockwise_divide(src, quanti):
    pass


def perform_blockwise_idct_tf(dct_img):
    shape = tf.shape(dct_img)
    x, y, c = shape[0], shape[1], shape[2]
    img_res = tf.reshape(dct_img, [x // 8, 8, y // 8, 8, c])
    img_dct1 = tf.signal.dct(tf.transpose(img_res, [0, 2, 3, 4, 1]), type=3, norm='ortho')
    img_dct2 = tf.signal.dct(tf.transpose(img_dct1, [0, 4, 1, 3, 2]), type=3, norm='ortho')
    out = tf.reshape(tf.transpose(img_dct2, [0, 1, 2, 4, 3]), shape)
    return out


def perform_blockwise_dct(img):
    imsize = img.shape
    dct_blocks = np.zeros(imsize, dtype=img.dtype)
    for i in np.r_[:imsize[0]:8]:
        for j in np.r_[:imsize[1]:8]:
            dct_blocks[i:(i + 8), j:(j + 8), 0] = dct(dct(img[i:(i + 8), j:(j + 8), 0].T, norm='ortho').T, norm='ortho')
            dct_blocks[i:(i + 8), j:(j + 8), 1] = dct(dct(img[i:(i + 8), j:(j + 8), 1].T, norm='ortho').T, norm='ortho')
            dct_blocks[i:(i + 8), j:(j + 8), 2] = dct(dct(img[i:(i + 8), j:(j + 8), 2].T, norm='ortho').T, norm='ortho')
    return dct_blocks


def perform_blockwise_idct(dct_img, dct_matrix=None):
    imsize = dct_img.shape
    dct_blocks = np.zeros(imsize, dtype=img.dtype)
    for i in np.r_[:imsize[0]:8]:
        for j in np.r_[:imsize[1]:8]:
            # dct_blocks[i:(i + 8), j:(j + 8), 0] = idct_transformer(dct_img[i:(i + 8), j:(j + 8), 0],dct_matrix)
            # dct_blocks[i:(i + 8), j:(j + 8), 1] = idct_transformer(dct_img[i:(i + 8), j:(j + 8), 1],dct_matrix)
            # dct_blocks[i:(i + 8), j:(j + 8), 2] = idct_transformer(dct_img[i:(i + 8), j:(j + 8), 2],dct_matrix)
            dct_blocks[i:(i + 8), j:(j + 8), 0] = idct(idct(dct_img[i:(i + 8), j:(j + 8), 0].T, norm='ortho').T,
                                                       norm='ortho')
            dct_blocks[i:(i + 8), j:(j + 8), 1] = idct(idct(dct_img[i:(i + 8), j:(j + 8), 1].T, norm='ortho').T,
                                                       norm='ortho')
            dct_blocks[i:(i + 8), j:(j + 8), 2] = idct(idct(dct_img[i:(i + 8), j:(j + 8), 2].T, norm='ortho').T,
                                                       norm='ortho')
    return dct_blocks


if __name__ == "__main__":
    img = cv2.imread("test.png")
    # img = cv2.imread("test1.jpg")
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    height = img.shape[0]
    width = img.shape[1]
    rectified_image_yuv = image_regulator(img_yuv, height, width)

    dct_matrix = dct_matrix_creator()
    # np.random.seed(100)
    # DCT in TensorFlow only supports float32
    rectified_image_yuv = np.random.rand(16, 32, 3).astype(np.float32)

    #     img = cv2.resize(img, None, fx=0.6, fy=0.6)
    #     dct_matrix = dct_matrix_creator()
    #     quantization_matrix_lc = np.load(os.path.join("quantization_tables", "Adobe_Photoshop__Save_As_00_lc.npy")).reshape(
    #         8, 8)
    #     quantization_matrix_cc = np.load(os.path.join("quantization_tables", "Adobe_Photoshop__Save_As_00_lc.npy")).reshape(
    #         8, 8)
    #     processed_img = background_processor(img, dct_matrix, quantization_matrix_lc, quantization_matrix_lc)
    #     print(processed_img.max())
    #     print(processed_img.min())

    '''
    out1 = perform_blockwise_dct(rectified_image_yuv.astype(np.float32))
    out2 = perform_blockwise_dct_tf(rectified_image_yuv.astype(np.float32))  # dct image
    numpy_output = out2.numpy()
    # idct = perform_blockwise_idct(numpy_output,dct_matrix)
    idct = perform_blockwise_idct_tf(out2).numpy().astype("uint8")
    idct = cv2.cvtColor(idct,cv2.COLOR_YUV2BGR)
    cv2.imwrite("testxxxxx.png", idct)
    # open("tensor_recovered.png", 'wb').write(png_data)
    # There is a bit of error
    print(np.allclose(out1, out2, rtol=1e-2, atol=1e-3))
'''
