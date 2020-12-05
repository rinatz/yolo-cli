#!/usr/bin/env python

from enum import Enum
import json

import rich
import typer

from yolo.constants import SCALE, SCORE_THRESHOLD, NMS_THRESHOLD
from yolo.model import load_model, Weights


def main(
    # fmt: off
    image: str = typer.Argument(..., help="Image path"),
    scale: float = typer.Option(SCALE, help="The scaling factor of mean subtraction"),
    weights: Weights = typer.Option(Weights.middle, "--weights", "-w", help="Model weights"),
    score_threshold: float = typer.Option(SCORE_THRESHOLD, help="The confidence score threshold"),
    nms_threshold: float = typer.Option(NMS_THRESHOLD, help="The non-maximum supression threshold"),
    names: str = typer.Option(None, "--names", "-n", help="Comma separated names to annotate"),
    output: str = typer.Option(None, "--output", "-o", help="Filename to store annotations"),
    output_image: str = typer.Option(None, help="Annotated image file to store"),
    pretty: bool = typer.Option(False, "--pretty", "-p", help="Print annotations with pretty-printed"),
    # fmt: on
):
    model = load_model()

    result = model.annotate(
        image,
        scale=scale,
        weights=weights,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        names=names.split(",") if names else [],
        output_image_path=output_image,
    )

    indent = 4 if pretty else None
    content = json.dumps(result, indent=indent, ensure_ascii=False)

    if output:
        with open(output, "w") as f:
            f.write(content)
    else:
        if pretty:
            rich.print(content)
        else:
            print(content)


if __name__ == "__main__":
    typer.run(main)
