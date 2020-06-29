import os
from SSIM_PIL import compare_ssim
from PIL import Image
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
    dct_matrix = dct_matrix_creator()
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
            class_ids, confidences, boxes, indexes = object_detector(net, output_layers, frame)
            font = cv2.FONT_HERSHEY_PLAIN
            frame_objects = []
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[i]
                    # cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    # cv2.putText(img, label, (x, y + 30), font, 1, color, 1)
                    identified_object = frame[y:(y + h), x:(x + w)]
                    frame_objects.append((identified_object, x, y, w, h, label))
            if frame_objects:
                processed_background = background_processor(frame, dct_matrix, quantization_matrix_lc,
                                                            quantization_matrix_cc)
                processed_background_bgr = cv2.cvtColor(processed_background, cv2.COLOR_YUV2BGR)
            else:
                processed_background_bgr = frame
            for item in frame_objects:
                processed_background_bgr[item[2]:(item[2] + item[4]), item[1]:(item[1] + item[3])] = item[0]
            display_img = processed_background_bgr
            cv2.imwrite(os.path.join("test_output", "processed", "processed%06d.png" % counter),
                        processed_background_bgr)
            cv2.imwrite(os.path.join("test_output", "original", "original%06d.png" % counter), frame)
            image1 = Image.open(os.path.join("test_output", "original", "original%06d.png" % counter))
            image2 = Image.open(os.path.join("test_output", "processed", "processed%06d.png" % counter))
            print("Processed %d Frame" % counter)
            ssim = compare_ssim(image1,image2)
            print("SSIM: "+str(ssim))
            cv2.putText(display_img,str(ssim),(0, 0 + 30), font, 1, colors[2], 1)
            cv2.imshow("Image", display_img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
