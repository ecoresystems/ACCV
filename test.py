import cv2
import numpy as np
from image_processor import *
import JpegEncoder

if __name__ == "__main__":
    counter = 0
    # Load YOLO
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # String Capturing process
    cap = cv2.VideoCapture('test_x264.mp4')
    if not cap.isOpened():
        print("Error opening video stream or file")

    while cap.isOpened():
        ret, frame = cap.read()
        counter += 1
        counterx = 0
        if ret:
            # img = cv2.resize(frame, None, fx=0.8, fy=0.8)
            img = frame
            b, g, r = cv2.split(img)
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            y_channel, u_channel, v_channel = cv2.split(img_yuv)
            class_ids, confidences, boxes, indexes = object_detector(net, output_layers, img)
            # y_channel_lowpassed = low_pass_filtering(y_channel, 200)
            # b_lowpassed = low_pass_filtering(b, 100)
            # g_lowpassed = low_pass_filtering(g, 100)
            # r_lowpassed = low_pass_filtering(r, 100)
            # restored_image = cv2.merge([b_lowpassed, g_lowpassed, r_lowpassed])
            font = cv2.FONT_HERSHEY_PLAIN
            frame_objects = []
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[i]
                    # cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    # cv2.putText(img, label, (x, y + 30), font, 1, color, 1)
                    identified_object = img[y:(y + h), x:(x + w)]
                    frame_objects.append((identified_object, x, y, w, h, label))
            if len(frame_objects) > 0:
                blured_background = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
            else:
                blured_background = frame
            for item in frame_objects:
                blured_background[item[2]:(item[2] + item[4]), item[1]:(item[1] + item[3])] = item[0]
            cv2.imshow("Image", blured_background)
            cv2.imwrite("test_output/blured%d.png" % counter, blured_background)
            cv2.imwrite("test_output/original%d.png" % counter, frame)
            # break
            # print(y_channel)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            pass
            # break
