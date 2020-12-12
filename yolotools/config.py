from pathlib import Path

# fmt: off
DATASET_PATH = Path("~/.yolotools").expanduser().absolute()
CONFIG_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
MODEL_URL = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
NAMES_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/coco.names"
# fmt : on

REGULARIZER = 1 / 255.0
SCORE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.3

BBOX_BORDER_COLOR = [0x63, 0x1E, 0xE9]
