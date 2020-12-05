#!/usr/bin/env python

import csv
from enum import Enum
import io
from pathlib import Path
import sys

import typer
import xmltodict


class RowSet(str, Enum):
    training = "TRAINING"
    validation = "VALIDATION"
    test = "TEST"
    unassigned = "UNASSIGNED"


def to_automl_csv(
    xml: dict, row_set: RowSet = RowSet.training, bucket: str = "", prefix: str = ""
):
    #
    # Formatting a training data CSV - Google Cloud Vision API
    # https://cloud.google.com/vision/automl/object-detection/docs/csv-format
    #
    out = io.StringIO()
    writer = csv.writer(out)

    annotation = xml["annotation"]

    path = Path(annotation["path"])
    size = annotation["size"]
    width = float(size["width"])
    height = float(size["width"])

    if bucket:
        path = Path(prefix).joinpath(path.name)
        uri = f"gs://{bucket}/{path}"
    else:
        uri = str(path)

    for object_ in annotation["object"]:
        name = object_["name"]

        bbox = object_["bndbox"]
        xmin = float(bbox["xmin"]) / width
        ymin = float(bbox["ymax"]) / height
        xmax = float(bbox["xmax"]) / width
        ymax = float(bbox["ymax"]) / height

        row = [
            row_set,
            uri,
            name,
            xmin,
            ymin,
            xmax,
            ymin,
            xmax,
            ymax,
            xmin,
            ymax,
        ]

        writer.writerow(row)

    return out.getvalue()


def main(
    # fmt: off
    xml_path: str = typer.Argument(None, help="Annotation file formatted with Pascal VOC"),
    row_set: RowSet = typer.Option(RowSet.training, "--row-set", "-s", help="Which set to assign rows in csv to"),
    bucket: str = typer.Option("", "--bucket", "-b", show_default=False, help="Bucket name of Google Storage"),
    prefix: str = typer.Option("", "--prefix", "-p", show_default=False, help="Prefix of Google Storage"),
    output: str = typer.Option(None, "--output", "-o", help="Filename to output csv"),
    # fmt: on
):
    if xml_path:
        with open(xml_path) as f:
            xml = xmltodict.parse(f.read(), encoding="utf-8", force_list=["object"])
    else:
        xml = xmltodict.parse(sys.stdin.read(), encoding="utf-8", force_list=["object"])

    rows = to_automl_csv(xml, row_set=row_set, bucket=bucket, prefix=prefix)

    if output:
        with open(output, "w", encoding="utf-8", newline="") as f:
            f.write(rows)
    else:
        # Not use print() to prevent writing last line
        sys.stdout.write(rows)


if __name__ == "__main__":
    typer.run(main)
