import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    vehicle_update_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        # Extract video properties
        self.video = cv2.VideoCapture('HSCC Interstate Highway Surveillance System - TEST VIDEO.mp4')
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frames_per_second = self.video.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize predictor
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
        self.predictor = DefaultPredictor(self.cfg)
        self.vehicle_number = 0
        self.current_number = 0
        self.previous_number = 0
        self.frame_count = 0

    def run(self):

        while self._run_flag:
            hasFrame, frame = self.video.read()
            if not hasFrame:
                break
            left = frame[0:self.height, 0:int(self.width / 2)]
            right = frame[0:self.height, int(self.width / 2):self.width]
            # Get prediction results for this frame

            left_roi = left[int(self.height / 3):self.height, 0:int(self.width / 2)]
            left_remaining = left[0:int(self.height / 3), 0:int(self.width / 2)]
            outputs = self.predictor(left_roi)
            outputs = outputs["instances"].to("cpu")
            pred_classes = outputs.pred_classes.tolist()
            pred_scores = outputs.scores.tolist()
            class_car = 2
            class_truck = 7
            vehicle_class = {class_car: "car", class_truck: "truck"}
            boxes = []
            confidences = []
            if len(outputs) == 0 or (class_car not in pred_classes and class_truck not in pred_classes):
                left = np.concatenate((left_remaining, left_roi), axis=0)

                result = np.concatenate((left, right), axis=1)

                self.change_pixmap_signal.emit(result)
                continue
            else:
                for i in range(len(outputs)):
                    score = pred_scores[i]
                    if score > 0.8:
                        x0, y0, x1, y1 = outputs.pred_boxes.tensor[i].tolist()
                        center_x = int(outputs.pred_boxes.get_centers()[i, 0].tolist())
                        center_y = int(outputs.pred_boxes.get_centers()[i, 1].tolist())
                        width = x1 - x0
                        height = y1 - y0
                        boxes.append((center_x, center_y, width, height))
                        confidences.append(score)

                multiTracker = cv2.MultiTracker_create()

                for bbox in boxes:
                    tracker = cv2.TrackerMOSSE_create()
                    multiTracker.add(tracker, left_roi, bbox)

                success, boxes = multiTracker.update(left_roi)
                print(boxes, len(boxes))

                if success:
                    for i, newbox in enumerate(boxes):
                        p1 = (int(newbox[0]) - 30, int(newbox[1]) - 20)
                        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                        cv2.putText(left_roi, vehicle_class[pred_classes[i]] + "  " + str(i + 1),
                                    (int(newbox[0]) - 10, int(newbox[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.rectangle(left_roi, p1, p2, (255, 128, 0), 2, 1)
                    self.current_number = len(boxes)
                    if self.frame_count % 30 == 0:
                        if self.current_number > self.previous_number:
                            self.vehicle_number += self.current_number
                            # self.previous_number = self.current_number
                            self.vehicle_update_signal.emit(self.vehicle_number)
                            self.previous_number = 0
                        else:
                            self.previous_number = self.current_number

                left = np.concatenate((left_remaining, left_roi), axis=0)

                result = np.concatenate((left, right), axis=1)

                self.change_pixmap_signal.emit(result)
            self.frame_count += 1

        self.video.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vehicle")
        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.textLabel = QLabel('Number of vehicles: 0')

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.vehicle_update_signal.connect(self.update_label)
        # start the thread
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def update_label(self, number):
        self.textLabel.setText("Number of vehicles: " + str(number))

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
