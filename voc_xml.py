#!/usr/bin/env python

import json
from pathlib import Path
import sys

from jinja2 import Template
import typer


XML_TEMPLATE = """<annotation>
    <folder>{{ path.parent.name }}</folder>
    <filename>{{ path.name }}</filename>
    <path>{{ path }}</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>{{ image["width"] }}</width>
        <height>{{ image["height"] }}</height>
        <depth>{{ image["channels"] }}</depth>
    </size>
    <segmented>0</segmented>
    {%- for annotation in annotations %}
    <object>
        <name>{{ annotation["name"] }}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{{ annotation["bbox"][0] }}</xmin>
            <ymin>{{ annotation["bbox"][1] }}</ymin>
            <xmax>{{ annotation["bbox"][2] }}</xmax>
            <ymax>{{ annotation["bbox"][3] }}</ymax>
        </bndbox>
    </object>
    {%- endfor %}
</annotation>
"""


def main(
    json_path: str = typer.Argument(None, help="Result of annotation"),
    output: str = typer.Option(None, "--output", "-o", help="Filename to output xml"),
):
    if json_path:
        with open(json_path, encoding="utf-8") as f:
            json_data = json.load(f)
    else:
        json_data = json.load(sys.stdin)

    template = Template(XML_TEMPLATE)
    image = json_data["image"]

    xml = template.render(
        path=Path(image["path"]),
        image=image,
        annotations=json_data["annotations"],
    )

    if output:
        with open(output, "w", encoding="utf-8", newline="") as f:
            f.write(xml)
    else:
        print(xml)


if __name__ == "__main__":
    typer.run(main)
