# `yolotools`

YOLO toolset for annotating images.

## Requirements

- Python 3
- Poetry

## Install

```shell
$ poetry install
$ poetry shell
```

## Object detection

Output annotations to stdout by JSON and write it to the image.

```shell
$ python -m yolo <IMAGE> --output-image output.jpg --pretty
```

When passing `--format=xml` option, resulting annotations are printed by Pascal VOC XML format.
