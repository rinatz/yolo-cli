# `yolotools`

YOLO toolset for annotating images.

## Requirements

- Python 3
- Poetry

## Install

```shell
$ pip install -U pip
$ pip install git+https://github.com/rinatz/yolotools
```

## Object detection

Output annotations to stdout by JSON and write it to the image.

```shell
$ yolo <IMAGE> --output-image output.jpg --pretty
```

When passing `--format=xml` option, resulting annotations are printed by Pascal VOC XML format.
