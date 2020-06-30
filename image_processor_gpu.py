import cv2
import numpy as np
import tensorflow as tf


def image_regulator(image, height, width):
    fill_height = height % 16
    fill_width = width % 16
    if fill_height != 0:
        fill_height = 16 - fill_height
    if fill_width != 0:
        fill_width = 16 - fill_width
    return np.pad(image, ((0, fill_height), (0, fill_width), (0, 0)), 'constant', constant_values=(0, 0))


def background_processor_gpu(img, quantization_matrix_lc, quantization_matrix_cc):
    height = img.shape[0]
    width = img.shape[1]
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    rectified_image_yuv = image_regulator(img_yuv, height, width)
    tensor_image = image_processor_gpu(rectified_image_yuv, quantization_matrix_lc, quantization_matrix_cc)
    return tensor_image.numpy().astype("uint8")[0:height, 0:width]


def image_processor_gpu(image_yuv, quantization_matrix_lc, quantization_matrix_cc):
    shape = tf.shape(image_yuv)
    idct = tf.zeros(shape, dtype=tf.float32)
    x, y, c = shape[0], shape[1], shape[2]
    shifted_image = tf.cast(image_yuv.astype(dtype="int16") - 128, dtype=tf.float32)
    quantization_matrix_lc = tf.reshape(quantization_matrix_lc, [8, 8])
    quantization_matrix_cc = tf.reshape(quantization_matrix_cc, [8, 8])
    quantization_matrix_lc_tile = tf.cast(tf.tile(quantization_matrix_lc, [x // 8, y // 8]), dtype=tf.float32)
    quantization_matrix_cc_tile = tf.cast(tf.tile(quantization_matrix_cc, [x // 8, y // 8]), dtype=tf.float32)
    dct_image = image_dct_transformer_gpu(shifted_image)
    quantified_dct_y_channel = tf.round(tf.divide(dct_image[:, :, 0], quantization_matrix_lc_tile))
    quantified_dct_u_channel = tf.round(tf.divide(dct_image[:, :, 1], quantization_matrix_cc_tile))
    quantified_dct_v_channel = tf.round(tf.divide(dct_image[:, :, 2], quantization_matrix_cc_tile))
    restored_dct_y_channel = tf.multiply(quantified_dct_y_channel, quantization_matrix_lc_tile)
    restored_dct_u_channel = tf.multiply(quantified_dct_u_channel, quantization_matrix_cc_tile)
    restored_dct_v_channel = tf.multiply(quantified_dct_v_channel, quantization_matrix_cc_tile)
    idct = image_idct_transformer_gpu(
        tf.stack([restored_dct_y_channel, restored_dct_u_channel, restored_dct_v_channel], -1))
    reshifted_image = tf.clip_by_value(idct + 128, clip_value_min=0, clip_value_max=255)
    return reshifted_image


def image_dct_transformer_gpu(img):
    shape = tf.shape(img)
    x, y, c = shape[0], shape[1], shape[2]
    img_res = tf.reshape(img, [x // 8, 8, y // 8, 8, c])
    img_dct1 = tf.signal.dct(tf.transpose(img_res, [0, 1, 2, 4, 3]), norm='ortho')
    img_dct2 = tf.signal.dct(tf.transpose(img_dct1, [0, 2, 4, 3, 1]), norm='ortho')
    out = tf.reshape(tf.transpose(img_dct2, [0, 4, 1, 2, 3]), shape)
    return out


def image_idct_transformer_gpu(dct_img):
    shape = tf.shape(dct_img)
    x, y, c = shape[0], shape[1], shape[2]
    img_res = tf.reshape(dct_img, [x // 8, 8, y // 8, 8, c])
    img_dct1 = tf.signal.dct(tf.transpose(img_res, [0, 2, 3, 4, 1]), type=3, norm='ortho')
    img_dct2 = tf.signal.dct(tf.transpose(img_dct1, [0, 4, 1, 3, 2]), type=3, norm='ortho')
    out = tf.reshape(tf.transpose(img_dct2, [0, 1, 2, 4, 3]), shape)
    return out
