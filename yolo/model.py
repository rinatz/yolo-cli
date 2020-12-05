from enum import Enum
from pathlib import Path
from typing import List, Union

import cv2 as cv
import numpy as np

from yolo.dataset import download

from yolo.constants import (
    DATASET_PATH,
    CONFIG_URL,
    MODEL_URL,
    NAMES_URL,
    SCALE,
    SCORE_THRESHOLD,
    NMS_THRESHOLD,
    BGR_PINK,
)


class Weights(str, Enum):
    small = "small"
    middle = "middle"
    large = "large"
    xlarge = "xlarge"

    @property
    def size(self):
        if self is Weights.small:
            return 320, 320
        elif self is Weights.middle:
            return 416, 416
        elif self is Weights.large:
            return 512, 512
        elif self is Weights.xlarge:
            return 608, 608


class Yolo:
    def __init__(self, dataset_path: Union[str, Path]):
        dataset_path = Path(dataset_path)
        config_path = str(dataset_path.joinpath("yolov4.cfg"))
        model_path = str(dataset_path.joinpath("yolov4.weights"))
        names_path = str(dataset_path.joinpath("coco.names"))

        self.net = cv.dnn.readNet(config_path, model_path)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

        with open(names_path) as f:
            self.names = f.read().splitlines()

    def annotate(
        self,
        image_path: Union[str, Path],
        *,
        scale: float = SCALE,
        weights: Weights = Weights.middle,
        score_threshold: float = SCORE_THRESHOLD,
        nms_threshold: float = NMS_THRESHOLD,
        names: List[str] = [],
        output_image_path: str = None,
    ):
        image_path = str(Path(image_path).absolute())

        img = cv.imread(image_path)
        img_height, img_width, img_channels = img.shape

        blob = cv.dnn.blobFromImage(img, scale, weights.size, swapRB=True, crop=False)

        self.net.setInput(blob)
        layer_names = self.net.getUnconnectedOutLayersNames()
        outputs = self.net.forward(layer_names)

        bboxes, labels, scores = [], [], []

        for output in np.vstack(outputs):
            label = np.argmax(output[5:])
            score = output[5:][label]

            if score <= score_threshold:
                continue

            if names and self.names[label] not in names:
                continue

            x, y, width, height = output[:4] * np.array(
                [img_width, img_height, img_width, img_height]
            )

            x_min, y_min = x - width // 2, y - height // 2

            bboxes.append([int(x_min), int(y_min), int(width), int(height)])
            labels.append(label)
            scores.append(float(score))

        annotations = []
        indices = cv.dnn.NMSBoxes(bboxes, scores, score_threshold, nms_threshold)

        if not len(indices):
            indices = np.array([])

        for i in indices.flatten():
            x_min, y_min, width, height = bboxes[i]
            x_max, y_max = x_min + width, y_min + height

            label, score = labels[i], scores[i]
            name = self.names[label]

            annotations.append(
                {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "name": name,
                    "score": score,
                }
            )

            if output_image_path:
                cv.rectangle(
                    img,
                    pt1=(x_min, y_min),
                    pt2=(x_max, y_max),
                    color=BGR_PINK,
                    thickness=4,
                )
                cv.putText(
                    img,
                    text=f"{name}: {score:.2f}",
                    org=(x_min, y_min - 20),
                    fontFace=cv.FONT_HERSHEY_DUPLEX,
                    fontScale=2,
                    color=BGR_PINK,
                    thickness=2,
                )

        if output_image_path:
            cv.imwrite(output_image_path, img)

        return {
            "image": {
                "path": image_path,
                "width": img_width,
                "height": img_height,
                "channels": img_channels,
            },
            "annotations": annotations,
        }


def load_model():
    download(CONFIG_URL, DATASET_PATH)
    download(MODEL_URL, DATASET_PATH)
    download(NAMES_URL, DATASET_PATH)

    return Yolo(DATASET_PATH)
