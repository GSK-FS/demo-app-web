from PIL import Image
import numpy as np
import cv2
import base64
#import imutils
#import easyocr
#from skimage.metrics import structural_similarity

def process_images(design_image_path, developer_image_path):
    try:
        design_image = cv2.imread(design_image_path)
        developer_image = cv2.imread(developer_image_path)

        if design_image is None or developer_image is None:
            raise ValueError("Error loading images.")

        # Your image processing logic goes here
        # For example, converting to grayscale
        gray_design_image = cv2.cvtColor(design_image, cv2.COLOR_BGR2GRAY)
        gray_developer_image = cv2.cvtColor(developer_image, cv2.COLOR_BGR2GRAY)

        # Return the processed images
        return gray_design_image, gray_developer_image
    except Exception as e:
        raise ValueError("Error processing images.") from e
