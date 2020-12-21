from pathlib import Path
from typing import List, Union

import cv2 as cv
import numpy as np

from yolotools.dataset import download

from yolotools.config import (
    DATASET_PATH,
    CONFIG_URL,
    MODEL_URL,
    NAMES_URL,
    SCORE_THRESHOLD,
    IOU_THRESHOLD,
    BOX_BORDER_COLOR,
)


class Yolo:
    def __init__(self, dataset_path: Union[str, Path] = DATASET_PATH):
        dataset_path = Path(dataset_path)

        config_path = download(CONFIG_URL, dataset_path)
        model_path = download(MODEL_URL, dataset_path)
        names_path = download(NAMES_URL, dataset_path)

        self._net = cv.dnn.readNet(str(config_path), str(model_path))
        self._net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self._input_size = (416, 416)

        with names_path.open() as f:
            self._names = {k: v.strip() for k, v in enumerate(f)}

    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, value):
        self._input_size = value

    @property
    def names(self):
        return self._names

    def predict(
        self,
        image,
        *,
        score_threshold: float = SCORE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        names: List[str] = [],
    ):
        x = cv.dnn.blobFromImage(
            image, 1 / 255.0, self.input_size, swapRB=True, crop=False
        )

        self._net.setInput(x)

        layer_names = self._net.getUnconnectedOutLayersNames()
        y = self._net.forward(layer_names)

        boxes, labels, scores = [], [], []

        for pred in np.vstack(y):
            label = np.argmax(pred[5:])
            score = pred[5:][label]

            if score <= score_threshold:
                continue

            x_center, y_center, width, height = pred[:4]
            x_min, y_min = x_center - width / 2.0, y_center - height / 2.0

            if x_min < 0.0:
                x_min = 0.0

            if y_min < 0.0:
                y_min = 0.0

            boxes.append([float(x_min), float(y_min), float(width), float(height)])
            labels.append(label)
            scores.append(float(score))

        indices = cv.dnn.NMSBoxes(boxes, scores, score_threshold, iou_threshold)

        predictions = []

        for i in indices.flatten():
            x_min, y_min, width, height = boxes[i]
            label, score = labels[i], scores[i]

            if names and self.names[label] not in names:
                continue

            predictions.append([x_min, y_min, width, height, label, score])

        return predictions

    def draw_bounding_boxes(self, image, predictions):
        output = np.copy(image)
        h, w = output.shape[:2]
        scale = np.array([w, h, w, h])

        for pred in predictions:
            x_min, y_min, width, height = pred[:4] * scale
            x_max, y_max = x_min + width, y_min + height

            name, score = self.names[pred[4]], pred[5]

            cv.rectangle(
                output,
                pt1=(int(x_min), int(y_min)),
                pt2=(int(x_max), int(y_max)),
                color=BOX_BORDER_COLOR,
                thickness=4,
            )
            cv.putText(
                output,
                text=f"{name}: {score * 100:.1f}%",
                org=(int(x_min), int(y_min) - 20),
                fontFace=cv.FONT_HERSHEY_DUPLEX,
                fontScale=2,
                color=BOX_BORDER_COLOR,
                thickness=2,
            )

        return output

    def infer(
        self,
        image_path,
        *,
        score_threshold: float = SCORE_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        names: List[str] = [],
    ):
        image = cv.imread(image_path)

        preds = self.predict(
            image,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            names=names,
        )

        output_image = self.draw_bounding_boxes(image, preds)

        return preds, output_image
