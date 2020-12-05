# `yolo-cli`

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
$ python -m yolo <IMAGE> --output-image output.jpg
```

## Output annotations by Pascal VOC format

```shell
$ python -m yolo <IMAGE> | python voc_xml.py
```

## Output annotations by Google Vision AutoML CSV format

```shell
$ python -m yolo <IMAGE> | python voc_xml.py | python automl_csv.py
```
