import numpy as np
import cv2


class YOLOv3_human:
    def __init__(self, width: int, height: int, threshold_probability: float = 0.6, threshold_iou: float = 0.4) -> None:
        ''' initialization of the network '''
        # prepare network
        self.net = cv2.dnn.readNetFromDarknet(
            './models/yolo/yolov3.cfg', './models/yolo/yolov3.weights')
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        # minimum probability to accept boudning box
        self.threshold_probability = threshold_probability
        # intersection over union
        self.threshold_iou = threshold_iou
        # layers
        self.ln = self.net.getLayerNames()
        self.output_ln = self.net.getUnconnectedOutLayersNames()
        # size
        self.width = width
        self.height = height

    def _bounding_boxes_calculations(self, net_output):
        ''' calulate bounding boxes from net output '''
        boxes = []
        confidences = []
        for output in net_output:
            for detection in output:
                probabilities = detection[5:]
                # extract highest probability class
                class_id = np.argmax(probabilities)
                if class_id != 0:
                    # not a person class (coco class names id 0 = human)
                    continue
                # extract value of highest probability
                confidence = probabilities[class_id]

                if confidence < self.threshold_probability:
                    continue

                # scale bounding box to image size
                # multiply the normalized YOLO coordinates by the width and height of the original image to get the actual pixel values
                box = detection[0:4] * \
                    np.array([self.width, self.height,
                             self.width, self.height])
                (center_x, center_y, width, height) = box
                # prepare bounding box to cv format of (x_min, y_min, width, height)
                x_min = int(center_x - (width / 2))
                y_min = int(center_y - (height / 2))
                # for optioal drawing later
                box = [x_min, y_min, int(width), int(height)]

                # append all
                boxes.append(box)
                confidences.append(float(confidence))
        return (boxes, confidences)

    def visualize_image(self, image, bboxes, confidences):
        ''' visualize given image with its detected human boudning boxes'''
        frame = image.copy()
        for i, box in enumerate(bboxes):
            x_min, y_min, w, h = box
            cv2.rectangle(frame, (x_min, y_min), (x_min+w, y_min+h),
                          color=(255, 0, 0), thickness=1)
            # text = "{}: {:.2f}".format('Human', confidences[i])
            # cv2.putText(frame, text, (x_min + 5, y_min - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return frame

    def detect(self, image):
        blob = cv2.dnn.blobFromImage(image, 1/255.0, swapRB=True, crop=False)
        # fowrard pass
        self.net.setInput(blob)
        out = self.net.forward(self.output_ln)
        # bounding boxes calculations
        boxes, confidences = self._bounding_boxes_calculations(out)
        # non maximum supression to filter out boxes
        result_ids = cv2.dnn.NMSBoxes(
            boxes, confidences, self.threshold_probability, self.threshold_iou)

        return np.array(boxes)[result_ids], np.array(confidences)[result_ids]
