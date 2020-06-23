import numpy as np
import cv2
from scipy.fft import dct, idct


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


def dct_transformer(block, dct_matrix):
    res = np.dot(dct_matrix, block)
    return np.dot(res, dct_matrix.T)


def channel_regulator(channel, height, width):
    fill_height = height % 16
    fill_width = width % 16
    if fill_height != 0:
        fill_height = 16 - fill_height
    if fill_width != 0:
        fill_width = 16 - fill_width
    return np.pad(channel, ((0, fill_height), (0, fill_width)), 'constant', constant_values=(0, 0))


def quantizer(block, quantization_matrix):
    return np.round(block / quantization_matrix.reshape(8, 8))


def dct2(a):
    return dct(dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(a):
    return idct(idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def low_pass_filtering(image, radius):
    # 对图像进行傅里叶变换，fft是一个三维数组，fft[:, :, 0]为实数部分，fft[:, :, 1]为虚数部分
    fft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 对fft进行中心化，生成的dshift仍然是一个三维数组
    dshift = np.fft.fftshift(fft)

    # 得到中心像素
    rows, cols = image.shape[:2]
    mid_row, mid_col = int(rows / 2), int(cols / 2)

    # 构建掩模，256位，两个通道
    mask = np.zeros((rows, cols, 2), np.float32)
    mask[mid_row - radius:mid_row + radius, mid_col - radius:mid_col + radius] = 1

    # 给傅里叶变换结果乘掩模
    fft_filtering = dshift * mask
    # 傅里叶逆变换
    ishift = np.fft.ifftshift(fft_filtering)
    image_filtering = cv2.idft(ishift)
    image_filtering = cv2.magnitude(image_filtering[:, :, 0], image_filtering[:, :, 1])
    # 对逆变换结果进行归一化（一般对图像处理的最后一步都要进行归一化，特殊情况除外）
    cv2.normalize(image_filtering, image_filtering, 0, 1, cv2.NORM_MINMAX)
    return image_filtering


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
