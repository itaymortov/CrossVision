import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import supervision as sv


class ObjectDetection:

    def checkTrafficLight2(self, frame, results):
        frame_height, frame_width = frame.shape[:2]
        polygon = np.array([
            [0, 0],
            [frame_width, 0],
            [frame_width, int(frame_height * 0.5)],
            [0, int(frame_height * 0.5)]
        ])
        xyxys = []
        confidences = []
        class_ids = []

        zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=(frame.shape[0], frame.shape[1]))
        for result in results:
            xyxys.append(result.boxes.xyxy.cpu().numpy())
            confidences.append(result.boxes.conf.cpu().numpy())
            class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

        xyxys = np.concatenate(xyxys, axis=0)
        confidences = np.concatenate(confidences, axis=0)
        class_ids = np.concatenate(class_ids, axis=0)

        # Setup detections for visualization
        detections = sv.Detections(
            xyxy=xyxys,
            confidence=confidences,
            class_id=class_ids,
        )
        detections = detections[detections.class_id == 9]
        zone.trigger(detections=detections)
        if zone.current_count > 0:
            return True
        return False

    def checkTrafficLight(self, frame, results):
        # Define the top 30% of the frame
        top_frame = sv.Rect(x=0, y=0, width=frame.shape[1], height=frame.shape[0] * 0.5)

        # Loop over the detection results and check if there is a traffic light object
        for result in results:
            boxes = result.boxes
            class_ids = boxes.cls
            # Check if there is a traffic light object
            if 2 in class_ids:
                # Loop over the bounding boxes and check if there is any overlap with the top 30% of the frame
                for box in boxes.xyxy:
                    # Convert the box coordinates to a Rect object
                    rect = sv.Rect(x=box[0], y=box[1], width=box[2] - box[0], height=box[3] - box[1])
                    # Check if there is any overlap with the top 30% of the frame
                    if rect.intersects(top_frame):
                        return True
        return False

    def __init__(self, capture_index):

        self.capture_index = capture_index

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model = self.load_model()

        self.CLASS_NAMES_DICT = self.model.model.names

        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)

    def load_model(self):

        model = YOLO("yolov8s.pt")  # load a pretrained YOLOv8n model
        model.fuse()

        return model

    def predict(self, frame):

        results = self.model(frame, save=False, device=0, show=False, classes=[2, 9], conf=0.4)
        return results

    def plot_bboxes(self, results, frame):

        xyxys = []
        confidences = []
        class_ids = []

        # Extract detections for Traffic light class
        for result in results:
            boxes = result.boxes.cpu().numpy()
            class_id = boxes.cls
            conf = boxes.conf
            xyxy = boxes.xyxy

            xyxys.append(result.boxes.xyxy.cpu().numpy())
            confidences.append(result.boxes.conf.cpu().numpy())
            class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

        xyxys = np.concatenate(xyxys, axis=0)
        confidences = np.concatenate(confidences, axis=0)
        class_ids = np.concatenate(class_ids, axis=0)

        # Setup detections for visualization
        detections = sv.Detections(
            xyxy=xyxys,
            confidence=confidences,
            class_id=class_ids,
        )

        # Format custom labels
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                       for _, confidence, class_id, tracker_id
                       in detections]

        # Annotate and display frame
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)

        return frame

    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:

            start_time = time()

            ret, frame = cap.read()
            assert ret
            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)
            check = self.checkTrafficLight2(frame, results)
            if check is True:
                print(1)
            else:
                print(0)
            self.checkTrafficLight2(frame, results)

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('CrossVision', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


detector = ObjectDetection('test.mp4')
detector()

