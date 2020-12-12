from pathlib import Path


def to_voc(img_file, img, annotations):
    img_path = Path(img_file).absolute()

    content = {
        "annotation": {
            "folder": img_path.parent.name,
            "filename": img_path.name,
            "path": str(img_path),
            "source": {
                "database": "Unknown",
            },
            "size": {
                "width": img.shape[1],
                "height": img.shape[0],
                "depth": img.shape[2],
            },
            "segmented": 0,
        },
    }

    objects = []

    for annotation in annotations:
        objects.append(
            {
                "name": annotation["name"],
                "pose": "Unspecified",
                "truncated": 0,
                "difficult": 0,
                "bndbox": {
                    "xmin": annotation["bbox"][0],
                    "ymin": annotation["bbox"][1],
                    "xmax": annotation["bbox"][2],
                    "ymax": annotation["bbox"][3],
                },
            }
        )

    content["annotation"]["object"] = objects

    return content
