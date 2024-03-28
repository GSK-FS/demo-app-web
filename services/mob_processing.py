#Import packages
import os
import cv2
import imutils
import numpy as np
from PIL import Image

import easyocr #ocr

#from skimage.measure import compare_ssim #old
from skimage.metrics import structural_similarity #updated


def process_images_mob(design_image_path, developer_image_path):
    print('process_images')

    output_directory = '/Users/home/Desktop/GSK/Computer-Vision/demo-app/processed'
    print( '18',output_directory)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)


    design_image_name = os.path.basename(design_image_path)
    developer_image_name = os.path.basename(developer_image_path)

    # Construct processed image paths with the desired format
    processed_path_1 = os.path.join(output_directory, 'processed_' + design_image_name)
    processed_path_2 = os.path.join(output_directory, 'processed_' + developer_image_name)


    img1 = cv2.imread(design_image_path)
    img1 = cv2.resize(img1, (300, 660))
    

    img2 = cv2.imread(developer_image_path)
    img2 = cv2.resize(img2, (300, 660))

    #Crop for responsive
    cropped_image1 = img1[40:630, 0:300]
    img1 = cropped_image1
    cropped_image2 = img2[30:620, 0:300]
    img2 = cropped_image2

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    (similar, diff) = structural_similarity(gray1, gray2, full=True)

    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    kernal = np.ones((10, 5), np.uint8)
    dilate = cv2.dilate(thresh, kernal, iterations=2)

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    img1_processed = img1.copy()
    img2_processed = img2.copy()

    for contour in contours:
        if cv2.contourArea(contour) > 1:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img1_processed, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(img2_processed, (x, y), (x + w, y + h), (0, 0, 255), 2)

            aText = cv2.rectangle(img1_processed, (x, y), (x + w, y + h), (0, 0, 255), 2)
            croppedImg1 = aText[y - 2:y + h + 2, x - 2:x + w + 2]

            if not croppedImg1.size == 0:
                grayText1 = cv2.cvtColor(croppedImg1, cv2.COLOR_BGR2GRAY)
                thresh1 = cv2.threshold(grayText1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                reader1 = easyocr.Reader(['en'])
                resultRaw1 = reader1.readtext(grayText1)
                print(resultRaw1)

            bText = cv2.rectangle(img2_processed, (x, y), (x + w, y + h), (0, 0, 255), 2)
            croppedImg2 = bText[y - 2:y + h + 2, x - 2:x + w + 2]

            if not croppedImg2.size == 0:
                grayText2 = cv2.cvtColor(croppedImg2, cv2.COLOR_BGR2GRAY)
                thresh2 = cv2.threshold(grayText2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    
    print( '77',processed_path_2)

    cv2.imwrite(processed_path_1, img1_processed)
    cv2.imwrite(processed_path_2, img2_processed)

    print(f"Processed images saved successfully to: {processed_path_1} and {processed_path_2}")

