import numpy as np
from numpy import r_
import cv2
from scipy.fft import dct, idct


def background_processor(img, dct_matrix, quantization_matrix_lc, quantization_matrix_cc):
    height = img.shape[0]
    width = img.shape[1]
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y_channel, u_channel, v_channel = cv2.split(img_yuv)
    rectified_y_channel = channel_regulator(y_channel, height, width)
    rectified_u_channel = channel_regulator(u_channel, height, width)
    rectified_v_channel = channel_regulator(v_channel, height, width)
    reshifted_y_channel = channel_processor(rectified_y_channel, quantization_matrix_lc, dct_matrix)
    reshifted_u_channel = channel_processor(rectified_u_channel, quantization_matrix_cc, dct_matrix)
    reshifted_v_channel = channel_processor(rectified_v_channel, quantization_matrix_cc, dct_matrix)
    restored_img = cv2.merge([reshifted_y_channel, reshifted_u_channel, reshifted_v_channel])
    return restored_img[0:height, 0:width]


def dct_matrix_creator():
    dct_matrix = np.zeros(shape=(8, 8))
    for i in range(8):
        if i == 0:
            c = np.sqrt(1 / 8)
        else:
            c = np.sqrt(2 / 8)
        for j in range(8):
            dct_matrix[i, j] = c * np.cos(np.pi * i * (2 * j + 1) / (2 * 8))
    return dct_matrix


def channel_processor(channel, quantization_matrix, dct_matrix):
    imsize = channel.shape
    idct = np.zeros(imsize)
    shifted_channel = channel.astype(dtype="int16") - 128
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            idct[i:(i + 8), j:(j + 8)] = block_processor(shifted_channel[i:(i + 8), j:(j + 8)], dct_matrix,
                                                         quantization_matrix).astype(dtype="int16")
    reshifted_channel = np.clip((idct + 127), 0, 255).astype(dtype="uint8")
    return reshifted_channel


def block_processor(block, dct_matrix, quantization_matrix):
    dct_block = dct_transformer(block, dct_matrix)
    quantified_block = np.round(dct_block / quantization_matrix).astype(dtype="int16")
    restored_dct_block = quantified_block * quantization_matrix
    restored_block = idct_transformer(restored_dct_block, dct_matrix)
    return restored_block


def dct_transformer(block, dct_matrix):
    res = np.dot(dct_matrix, block)
    return np.dot(res, dct_matrix.T)


def idct_transformer(block, dct_matrix):
    res = np.dot(dct_matrix.T, block)
    return np.dot(res, dct_matrix)


def channel_regulator(channel, height, width):
    fill_height = height % 16
    fill_width = width % 16
    if fill_height != 0:
        fill_height = 16 - fill_height
    if fill_width != 0:
        fill_width = 16 - fill_width
    return np.pad(channel, ((0, fill_height), (0, fill_width)), 'constant', constant_values=(0, 0))


def quantizer(block, quantization_matrix):
    return np.round(block / quantization_matrix.reshape(8, 8)).astype(int)


def dct2(a):
    return dct(dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(a):
    return idct(idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


# Load Yolo
def object_detector(net, output_layers, img):
    height, width, channels = img.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (412, 412), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return class_ids, confidences, boxes, indexes
    pass
