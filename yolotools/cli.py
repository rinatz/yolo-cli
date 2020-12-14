import sys

import cv2 as cv
import typer

from yolotools.config import SCORE_THRESHOLD, IOU_THRESHOLD
from yolotools.model import Yolo


def run(
    # fmt: off
    image_path: str = typer.Argument(..., help="Image path"),
    score_threshold: float = typer.Option(SCORE_THRESHOLD, help="The confidence score threshold"),
    iou_threshold: float = typer.Option(IOU_THRESHOLD, help="The non-maximum supression threshold"),
    names: str = typer.Option(None, "--names", "-n", help="Comma separated names to annotate"),
    output_path: str = typer.Option(None, "--output", "-o", help="Filename to store annotations"),
    output_image_path: str = typer.Option(None, "--output-image", help="Annotated image file to store"),
    # fmt: on
):
    yolo = Yolo()
    image = cv.imread(image_path)

    preds = yolo.predict(
        image,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        names=names,
    )

    if output_path:
        out = open(output_path, "w")
        close_output = out.close
    else:
        out = sys.stdout
        close_output = lambda: None

    try:
        for pred in preds:
            x, y, w, h, label, score = pred
            out.write(f"{label} {x:.06f} {y:.06f} {w:.06f} {h:.06f} {score:.06f}\n")
    finally:
        close_output()

    if output_image_path:
        output_image = yolo.plot_bbox(image, preds)
        cv.imwrite(output_image_path, output_image)


def main():
    typer.run(run)
