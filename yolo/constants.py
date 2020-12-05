from pathlib import Path

# fmt: off
DATASET_PATH = Path("~/.yolo-cli").expanduser().absolute()
CONFIG_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
MODEL_URL = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
NAMES_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/coco.names"

SCALE = 1 / 255.0
SCORE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.3

BGR_PINK = [0x63, 0x1E, 0xE9]
# fmt : on
