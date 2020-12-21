from pathlib import Path

# fmt: off
DATASET_PATH = Path("~/.yolotools").expanduser().absolute()
CONFIG_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
MODEL_URL = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
NAMES_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/coco.names"
# fmt : on

SCORE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.3

BOX_BORDER_COLOR = [0x63, 0x1E, 0xE9]
