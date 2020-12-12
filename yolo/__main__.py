#!/usr/bin/env python

from enum import Enum
import json

import cv2 as cv
import rich
import typer
import xmltodict

from yolo.config import REGULARIZER, SCORE_THRESHOLD, NMS_THRESHOLD
from yolo.model import (
    Weight,
    Yolo,
    load_img,
    img_to_array,
    decode_predictions,
    plot_bbox,
)
from yolo.voc import to_voc


class OutputFormat(str, Enum):
    json = "json"
    xml = "xml"


def main(
    # fmt: off
    image: str = typer.Argument(..., help="Image path"),
    regularizer: float = typer.Option(REGULARIZER, help="The regularization factor of mean subtraction"),
    weight: Weight = typer.Option(Weight.middle, "--weight", "-w", help="The scale of the model"),
    score_threshold: float = typer.Option(SCORE_THRESHOLD, help="The confidence score threshold"),
    nms_threshold: float = typer.Option(NMS_THRESHOLD, help="The non-maximum supression threshold"),
    names: str = typer.Option(None, "--names", "-n", help="Comma separated names to annotate"),
    pretty: bool = typer.Option(False, "--pretty", "-p", help="Print annotations with pretty-printed"),
    format: OutputFormat = typer.Option(OutputFormat.json, "--format", "-f", help="Output format of annotations"),
    output: str = typer.Option(None, "--output", "-o", help="Filename to store annotations"),
    output_image: str = typer.Option(None, help="Annotated image file to store"),
    # fmt: on
):
    model = Yolo()

    img = load_img(image)
    height, width = img.shape[:2]

    x = img_to_array(img, regularizer=regularizer, weight=weight)
    preds = model.predict(x)

    annotations = decode_predictions(
        preds,
        target_size=(width, height),
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        names=names.split(",") if names else [],
    )

    voc_dict = to_voc(image, img, annotations)

    if format == OutputFormat.json:
        indent = 4 if pretty else None
        content = json.dumps(voc_dict, indent=indent, ensure_ascii=False)
    elif format == OutputFormat.xml:
        content = xmltodict.unparse(voc_dict, pretty=pretty)
        content = content.replace('<?xml version="1.0" encoding="utf-8"?>', "").strip()
    else:
        raise typer.Exit(1)

    if output:
        with open(output, "w") as f:
            f.write(content)
    else:
        if pretty:
            rich.print(content)
        else:
            print(content)

    if output_image:
        out_img = plot_bbox(img, annotations)
        cv.imwrite(output_image, out_img)


if __name__ == "__main__":
    typer.run(main)
