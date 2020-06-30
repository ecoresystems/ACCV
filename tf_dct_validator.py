import numpy as np
import tensorflow as tf
from scipy.fftpack import dct

@tf.function
def perform_blockwise_dct_tf(img):
    shape = tf.shape(img)
    x, y = shape[0], shape[1]
    img_res = tf.reshape(img, [x // 8, 8, y // 8, 8])
    print(img_res)
    # img_dct1 = tf.spectral.dct(tf.transpose(img_res, [0, 1, 2, 4, 3]), norm='ortho')
    # img_dct2 = tf.spectral.dct(tf.transpose(img_dct1, [0, 2, 4, 3, 1]), norm='ortho')
    img_dct1 = tf.signal.dct(tf.transpose(img_res, [0, 1, 3,  2]), norm='ortho')
    img_dct2 = tf.signal.dct(tf.transpose(img_dct1, [0, 2, 3, 1]), norm='ortho')
    out = tf.reshape(tf.transpose(img_dct2, [0,  3, 1, 2]), shape)
    return out


def perform_blockwise_dct(img):
    imsize = img.shape
    dct_blocks = np.zeros(imsize, dtype=img.dtype)
    for i in np.r_[:imsize[0]:8]:
        for j in np.r_[:imsize[1]:8]:
            dct_blocks[i:(i + 8), j:(j + 8)] = dct(dct(img[i:(i + 8), j:(j + 8)].T, norm='ortho').T, norm='ortho')
    return dct_blocks

if __name__ == "__main__":
    np.random.seed(100)
    # DCT in TensorFlow only supports float32
    img = np.random.rand(128, 256).astype(np.float32)
    out1 = perform_blockwise_dct(img)
    out2 = perform_blockwise_dct_tf(img)
    # There is a bit of error
    print(np.allclose(out1, out2, rtol=1e-5, atol=1e-6))
