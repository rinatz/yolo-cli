from enum import Enum
from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import numpy as np

from yolotools.dataset import download

from yolotools.config import (
    DATASET_PATH,
    CONFIG_URL,
    MODEL_URL,
    NAMES_URL,
    REGULARIZER,
    SCORE_THRESHOLD,
    NMS_THRESHOLD,
    BBOX_BORDER_COLOR,
)


class Yolo:
    def __init__(self):
        download(CONFIG_URL, DATASET_PATH)
        download(MODEL_URL, DATASET_PATH)

        dataset_path = Path(DATASET_PATH)
        config_path = str(dataset_path.joinpath("yolov4.cfg"))
        model_path = str(dataset_path.joinpath("yolov4.weights"))

        self.net = cv.dnn.readNet(config_path, model_path)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

    def predict(self, x):
        self.net.setInput(x)
        layer_names = self.net.getUnconnectedOutLayersNames()

        preds = self.net.forward(layer_names)

        return preds


class Weight(str, Enum):
    small = "small"
    middle = "middle"
    large = "large"
    xlarge = "xlarge"


def load_img(img_file):
    return cv.imread(img_file)


def img_to_array(
    img, *, regularizer: float = REGULARIZER, weight: Weight = Weight.middle
):
    target_size = {
        Weight.small: (320, 320),
        Weight.middle: (416, 416),
        Weight.large: (512, 512),
        Weight.xlarge: (608, 608),
    }[weight]

    return cv.dnn.blobFromImage(img, regularizer, target_size, swapRB=True, crop=False)


def decode_predictions(
    preds,
    *,
    target_size: Tuple[int, int] = (1, 1),
    score_threshold: float = SCORE_THRESHOLD,
    nms_threshold: float = NMS_THRESHOLD,
    names: List[str] = [],
):
    download(NAMES_URL, DATASET_PATH)
    names_path = str(Path(DATASET_PATH).joinpath("coco.names"))

    with open(names_path) as f:
        all_names = f.read().splitlines()

    labels, scores, bboxes = [], [], []

    for pred in np.vstack(preds):
        label = np.argmax(pred[5:])
        score = pred[5:][label]

        if names and all_names[label] not in names:
            continue

        x, y, width, height = pred[:4] * np.array([*target_size, *target_size])
        x_min, y_min = x - width / 2.0, y - height / 2.0

        labels.append(label)
        scores.append(float(score))
        bboxes.append([int(x_min), int(y_min), int(width), int(height)])

    indices = cv.dnn.NMSBoxes(bboxes, scores, score_threshold, nms_threshold)

    if not len(indices):
        return []

    annotations = []

    for i in indices.flatten():
        x_min, y_min, width, height = bboxes[i]
        x_max, y_max = x_min + width, y_min + height

        if x_min < 0:
            x_min = 0

        if y_min < 0:
            y_min = 0

        if x_max > target_size[0]:
            x_max = target_size[0]

        if y_max > target_size[1]:
            y_max = target_size[1]

        label, score = labels[i], scores[i]
        name = all_names[label]

        annotations.append(
            {
                "name": name,
                "score": score,
                "bbox": [x_min, y_min, x_max, y_max],
            }
        )

    return annotations


def plot_bbox(img, annotations):
    out_img = np.copy(img)

    for annotation in annotations:
        x_min, y_min, x_max, y_max = annotation["bbox"]
        name = annotation["name"]
        score = annotation["score"]

        cv.rectangle(
            out_img,
            pt1=(x_min, y_min),
            pt2=(x_max, y_max),
            color=BBOX_BORDER_COLOR,
            thickness=4,
        )
        cv.putText(
            out_img,
            text=f"{name}: {score:.2f}",
            org=(x_min, y_min - 20),
            fontFace=cv.FONT_HERSHEY_DUPLEX,
            fontScale=2,
            color=BBOX_BORDER_COLOR,
            thickness=2,
        )

    return out_img
