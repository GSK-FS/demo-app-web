import os
import json
from PIL import Image, ImageDraw
import PIL

import cv2
import numpy as np
import math
from typing import Union

# from IPython.display import display

# import matplotlib.pyplot as plt

# import pytesseract

json_path = (
    "/Users/home/Desktop/GSK/Computer-Vision/yolo-test/custom-data/json-design.json"
)
design_image_path = "/Users/home/Desktop/GSK/Computer-Vision/yolo-test/custom-data/Power AgencyWeb Design.png"
developed_image_path = "/Users/home/Desktop/GSK/Computer-Vision/yolo-test/custom-data/PowerAgencyWeb_Developed.png"

# Load JSON data
with open(f"{json_path}") as f:
    data_json: dict = json.load(f)