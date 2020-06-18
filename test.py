import cv2
import numpy as np
from fram_processor import object_detector

if __name__ == "__main__":
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
        if ret:
            img = cv2.resize(frame, None, fx=0.8, fy=0.8)
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            y_channel, u_channel, v_channel = cv2.split(img_yuv)
            class_ids, confidences, boxes, indexes = object_detector(net, output_layers, img)
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[i]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 1, color, 1)
            y_channel_fft = np.fft.fft2(y_channel)
            y_channel_fft_spectrum = 20*np.log(np.abs(y_channel_fft))
            cv2.imshow("Image", y_channel_fft_spectrum)
            # print(y_channel)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
