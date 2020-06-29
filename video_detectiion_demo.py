import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from absl import app, flags
from absl.flags import FLAGS

import core.utils as utils
from core.config import cfg
from core.yolov4 import YOLOv4, decode


def run_recognition():
    STRIDES = np.array(cfg.YOLO.STRIDES)
    ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, False)
    XYSCALE = cfg.YOLO.XYSCALE
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    input_size = 608
    image_path = './data/kite.jpg'

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    feature_maps = YOLOv4(input_layer, NUM_CLASS)
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, NUM_CLASS, i)
        bbox_tensors.append(bbox_tensor)
    model = tf.keras.Model(input_layer, bbox_tensors)
    utils.load_weights(model, weights_file='yolov4.weights')
    model.summary()
    pred_bbox = model.predict(image_data)
    pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
    bboxes = utils.nms(bboxes, 0.213, method='nms')
    image = utils.draw_bbox(original_image, bboxes)
    image = Image.fromarray(image)
    image.show()



if __name__ == '__main__':
    run_recognition()
