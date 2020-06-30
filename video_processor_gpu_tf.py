import os
import time

from PIL import Image
from SSIM_PIL import compare_ssim

import core.utils as utils
from core.config import cfg
from core.yolov4 import YOLOv4, decode
from image_processor_gpu import *

if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)
    STRIDES = np.array(cfg.YOLO.STRIDES)
    ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, False)
    XYSCALE = cfg.YOLO.XYSCALE
    classes = utils.read_class_names(cfg.YOLO.CLASSES)
    NUM_CLASS = len(classes)
    input_size = 608
    counter = 0
    colors = np.random.uniform(0, 255, size=(NUM_CLASS, 3))
    # Initialize Yolov4
    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    feature_maps = YOLOv4(input_layer, NUM_CLASS)
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, NUM_CLASS, i)
        bbox_tensors.append(bbox_tensor)
    model = tf.keras.Model(input_layer, bbox_tensors)
    utils.load_weights(model, weights_file='yolov4.weights')
    # model.summary()

    # String Capturing process
    cap = cv2.VideoCapture('test_x264.mp4')
    if not cap.isOpened():
        print("Error opening video stream or file")
    quantization_matrix_lc = np.load(
        os.path.join("quantization_tables", "Adobe_Photoshop__Save_As_03_lc.npy")).reshape(
        8, 8)
    quantization_matrix_cc = np.load(
        os.path.join("quantization_tables", "Adobe_Photoshop__Save_As_03_cc.npy")).reshape(
        8, 8)
    while cap.isOpened():
        ret, frame = cap.read()
        counter += 1
        if ret:
            # img = cv2.resize(frame, None, fx=0.8, fy=0.8)
            image_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_size = frame.shape[:2]
            image_data = utils.image_preprocess(np.copy(image_data), [input_size, input_size])
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            prev_time = time.time()
            pred_bbox = model.predict(image_data)
            pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
            bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.25)
            bboxes = utils.nms(bboxes, 0.213, method='nms')
            font = cv2.FONT_HERSHEY_PLAIN
            frame_objects = []
            for bbox in bboxes:
                coor = np.array(bbox[:4], dtype="int32")
                score = bbox[4]
                class_index = int(bbox[5])
                # coor[0-4]= x, y, x+w, y+h respectively
                identified_object = frame[coor[1]:coor[3], coor[0]:coor[2]]
                frame_objects.append((identified_object, coor[0], coor[1], coor[2], coor[3], classes[class_index]))
            # image = utils.draw_bbox(frame, bboxes)
            if frame_objects:
                processed_background = background_processor_gpu(frame,quantization_matrix_lc,quantization_matrix_cc)
                processed_background_bgr = cv2.cvtColor(processed_background, cv2.COLOR_YUV2BGR)
            else:
                processed_background_bgr = frame
            for item in frame_objects:
                processed_background_bgr[item[2]:item[4], item[1]:item[3]] = item[0]
            display_img = processed_background_bgr
            cv2.imwrite(os.path.join("test_output", "processed", "processed%06d.png" % counter),
                        processed_background_bgr)
            cv2.imwrite(os.path.join("test_output", "original", "original%06d.png" % counter), frame)
            # image1 = Image.open(os.path.join("test_output", "original", "original%06d.png" % counter))
            # image2 = Image.open(os.path.join("test_output", "processed", "processed%06d.png" % counter))
            # print("Processed %d Frame" % counter)
            # ssim = compare_ssim(image1, image2)
            # print("SSIM: " + str(ssim))
            # cv2.putText(display_img, str(ssim), (0, 0 + 30), font, 1, colors[2], 1)
            cv2.imshow("Image", display_img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
