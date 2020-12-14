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
    BBOX_BORDER_COLOR,
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
        REGULARIZER = 1 / 255.0

        x = cv.dnn.blobFromImage(
            image, REGULARIZER, self.input_size, swapRB=True, crop=False
        )

        self._net.setInput(x)

        layer_names = self._net.getUnconnectedOutLayersNames()
        y = self._net.forward(layer_names)

        bboxes, labels, scores = [], [], []
        SCALE = np.array([100, 100, 100, 100], dtype=np.float)

        for pred in np.vstack(y):
            label = np.argmax(pred[5:])
            score = pred[5:][label]

            x_center, y_center, width, height = pred[:4] * SCALE
            x_min, y_min = x_center - width / 2.0, y_center - height / 2.0

            if x_min < 0.0:
                x_min = 0.0

            if y_min < 0.0:
                y_min = 0.0

            if x_min + width > image.shape[1]:
                width = image.shape[1] - x_min

            if y_min + height > image.shape[0]:
                height = image.shape[0] - y_min

            bboxes.append([int(x_min), int(y_min), int(width), int(height)])
            labels.append(label)
            scores.append(float(score))

        indices = cv.dnn.NMSBoxes(bboxes, scores, score_threshold, iou_threshold)

        if not len(indices):
            return []

        predictions = []

        for i in indices.flatten():
            x_min, y_min, width, height = bboxes[i] / SCALE
            x_center, y_center = x_min + width / 2.0, y_min + height / 2.0

            label, score = labels[i], scores[i]

            if names and self.names[label] not in names:
                continue

            predictions.append([x_center, y_center, width, height, label, score])

        return predictions

    def plot_bbox(self, image, predictions):
        output = np.copy(image)
        h, w = output.shape[:2]
        scale = np.array([w, h, w, h])

        for pred in predictions:
            x_center, y_center, width, height = pred[:4] * scale
            x_min, y_min = int(x_center - width / 2.0), int(y_center - height / 2.0)
            x_max, y_max = int(x_min + width), int(y_min + height)

            name, score = self.names[pred[4]], pred[5]

            cv.rectangle(
                output,
                pt1=(x_min, y_min),
                pt2=(x_max, y_max),
                color=BBOX_BORDER_COLOR,
                thickness=4,
            )
            cv.putText(
                output,
                text=f"{name}: {score * 100:.1f}%",
                org=(x_min, y_min - 20),
                fontFace=cv.FONT_HERSHEY_DUPLEX,
                fontScale=2,
                color=BBOX_BORDER_COLOR,
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

        output_image = self.plot_bbox(image, preds)

        return preds, output_image
