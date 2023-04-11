import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import supervision as sv

from ultralytics.yolo.utils.metrics import bbox_iou


class ObjectDetection:

    def __init__(self, capture_index):

        self.capture_index = capture_index

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model, self.model2 = self.load_models()

        # self.CLASS_NAMES_DICT = self.model.model.names
        self.CLASS_NAMES_DICT = {1: 'crosswalk', 2: 'car', 9: 'traffic light'}

        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=1, text_thickness=1, text_scale=0.4)

        self.GREEN = (0, 255, 0)  # Green for when is_under is True
        self.RED = (0, 0, 255)  # Red for when is_under is False
        self.color = self.GREEN

    def load_models(self):

        model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n model
        model2 = YOLO("best.pt")  # load our self trained model.
        model.fuse()
        model2.fuse()

        return model, model2

    def predict(self, frame):

        results_car_traffic = self.model(frame, save=False, device=0, show=False, classes=[2, 9], conf=0.4)
        # results_traffic = self.model(frame, save=False, device=0, show=False, classes=[9], conf=0.4)
        results2_crosswalk = self.model2(frame, save=False, device=0, show=False, classes=[1], conf=0.2)
        return results_car_traffic, results2_crosswalk

    def plot_bboxes(self, results, results2, frame):

        xyxys = []
        confidences = []
        class_ids = []
        boxes = {}
        boxes["cross"] = []
        boxes["car"] = []
        boxes["trafficlight"] = []

        for result in results:
            for res in result:

                xy = res.boxes.xyxy.cpu().numpy()
                conf = res.boxes.conf.cpu().numpy()
                id = res.boxes.cls.cpu().numpy().astype(int)
                if res.boxes.cls.cpu().numpy().astype(int) == 2:
                    xyxys.append(xy)
                    confidences.append(conf)
                    class_ids.append(id)
                    boxes["car"].append(xy.tolist()[0])
                if res.boxes.cls.cpu().numpy().astype(int) == 9:
                    xyxys.append(xy)
                    confidences.append(conf)
                    class_ids.append(id)
                    boxes["trafficlight"].append(xy.tolist()[0])

        for result2 in results2:
            for res2 in result2:
                xy = res2.boxes.xyxy.cpu().numpy()
                conf = res2.boxes.conf.cpu().numpy()
                id = res2.boxes.cls.cpu().numpy().astype(int)
                xyxys.append(xy)
                confidences.append(conf)
                class_ids.append(id)
                boxes["cross"].append(xy.tolist()[0])



        xyxys = np.concatenate(xyxys, axis=0)
        confidences = np.concatenate(confidences, axis=0)
        class_ids = np.concatenate(class_ids, axis=0)
        # Setup detections for visualization

        detections = sv.Detections(
            xyxy=xyxys,
            confidence=confidences,
            class_id=class_ids,
        )

        # if np.any(boxes["cross"]) and np.any(boxes["car"]) and not np.any(boxes["trafficlight"]):
        for boxcross in boxes["cross"]:
            for boxcar in boxes["car"]:
                if not self.checkTrafficLight(frame,detections):
                    print(self.is_under(boxcar, boxcross, 20))
                else:
                    self.color = self.GREEN

        # Format custom labels
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                       for _, confidence, class_id, tracker_id
                       in detections]

        # Annotate and display frame
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
        return frame

    def is_under(self, xyxy1, xyxy2, deviation):
        # Calculate the vertical overlap between the two bounding boxes
        y_overlap = min(xyxy1[3], xyxy2[3]) - max(xyxy1[1], xyxy2[1])

        # Calculate the vertical distance between the centers of the two bounding boxes
        y_distance = abs((xyxy1[1] + xyxy1[3]) / 2 - (xyxy2[1] + xyxy2[3]) / 2)

        # Check if the vertical overlap is positive and the vertical distance is within the deviation threshold
        if y_overlap > 0 and y_distance <= deviation:
            self.color = self.GREEN
            return False
        else:
            self.color = self.RED
            return True

    def checkTrafficLight(self, frame, detections):
        frame_height, frame_width = frame.shape[:2]
        polygon = np.array([
            [0, 0],
            [frame_width, 0],
            [frame_width, int(frame_height * 0.4)],
            [0, int(frame_height * 0.4)]
        ])
        zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=(frame.shape[0], frame.shape[1]))

        detections = detections[detections.class_id == 9]
        zone.trigger(detections=detections)
        if zone.current_count > 0:
            return True
        return False

    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        # cap.set(cv2.CAP_PROP_FPS, 5)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        width = cap.get(3)  # float `width`
        height = cap.get(4)  # float `height`

        while True:

            start_time = time()

            ret, frame = cap.read()
            if not ret:
                break
            results, results3 = self.predict(frame)
            frame = self.plot_bboxes(results, results3, frame)
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.rectangle(frame, (0, 0), (int(width), int(height)), self.color, 10)
            cv2.imshow('CrossVision', frame)
            # cv2.waitKey(100)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.waitKey(1000000)
        cap.release()
        cv2.destroyAllWindows()


detector = ObjectDetection('test3.mp4')
detector()
