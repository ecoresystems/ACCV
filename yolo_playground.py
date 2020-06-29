import time

import cv2
import numpy as np
import tensorflow as tf
from yolov4.tf import YOLOv4

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

yolo = YOLOv4()

yolo.classes = "data/classes/coco.names"

yolo.make_model()
yolo.load_weights("yolov4.weights", weights_type="yolo")

media_path = "test_x264.mp4"
vid = cv2.VideoCapture(media_path)
while True:
    return_value, frame = vid.read()
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("No image! Try with another video format")

    prev_time = time.time()
    image = yolo.predict(frame)
    curr_time = time.time()
    exec_time = curr_time - prev_time
    result = np.asarray(image)
    info = "time: %.2f ms" % (1000 * exec_time)
    print(info)
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("result", result)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
