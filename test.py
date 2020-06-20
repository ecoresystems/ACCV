import cv2
import numpy as np
from image_processor import *

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
        if ret and counter > 100:
            # img = cv2.resize(frame, None, fx=0.8, fy=0.8)
            img = frame
            b, g, r = cv2.split(img)
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
            # y_channel_fft = np.fft.fft2(y_channel)
            # y_channel_fft_spectrum = 20*np.log(np.abs(y_channel_fft))
            # f_ishift = np.fft.ifftshift(y_channel_fft)
            # img_back = cv2.idft(f_ishift)
            y_channel_lowpassed = low_pass_filtering(y_channel, 200)
            b_lowpassed = low_pass_filtering(b, 100)
            g_lowpassed = low_pass_filtering(g, 100)
            r_lowpassed = low_pass_filtering(r, 100)
            restored_image = cv2.merge([b_lowpassed, g_lowpassed, r_lowpassed])
            # print(np.max(y_channel_lowpassed))
            # print(np.min(y_channel_lowpassed))
            # print(img_yuv)
            # print((y_channel))
            # print((y_channel_lowpassed*255).astype(int).shape)
            # print(np.max(y_channel))
            # print(np.min(y_channel))
            # print(y_channel_lowpassed.shape)
            # print(u_channel.shape)
            # print(v_channel.shape)
            # restored_img_yuv = np.dstack([(y_channel_lowpassed).astype(int),u_channel,v_channel])
            # restored_img_yuv = cv2.merge([(y_channel_lowpassed*255).astype(int),u_channel,v_channel])

            # restored_img = cv2.cvtColor(restored_img_yuv,cv2.COLOR_YUV2BGR)
            cv2.imshow("Image", restored_image)
            # cv2.imwrite("output/img%d.png" % counter, y_channel_lowpassed*255)
            print(b_lowpassed)

            # print(y_channel)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            pass
            # break
