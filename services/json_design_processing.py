# services/json_design_processing.py

#Import packages
import os
import json
import cv2
# import imutils
import numpy as np
import PIL
from PIL import Image, ImageDraw
import math
from typing import Union


import easyocr #ocr

#from skimage.measure import compare_ssim #old
from skimage.metrics import structural_similarity #updated


def transform_accomulated(transform_0: dict, transform_1: dict) -> dict:
    transform_acc: dict = {}
    for key in transform_0.keys():
        if key in ["m02", "m12"]:
            transform_acc[key] = transform_0[key] + transform_1[key]
        else:
            transform_acc[key] = (
                transform_0[key] if transform_0[key] != 0 else transform_1[key]
            )
    return transform_acc

def rotation_factor(transform_0: dict) -> float:
    scale_x = transform_0["m00"]
    scale_y = transform_0["m10"]

    tan_x = transform_0["m01"]
    tan_y = transform_0["m11"]

    scale = (scale_x + scale_y) / 2
    rotation_radians = math.atan2(tan_x, tan_y)
    rotation_degrees = math.degrees(rotation_radians)

    rounded_rotation_degrees= round(rotation_degrees)
    return rotation_degrees

def rectangle_rotation(input_image_cv:np.ndarray, angle_pt, pt_1, pt_2, line_color, thickness:int=..., line_type:int=..., shift:int=...):
    print("rectangle_rotation")

    large_image_0 = input_image_cv
    # pt_1 = (20, 10) # x , y
    # pt_2 = (50, 50) # height, width
    x1, y1 = (int(pt_1[0]), int(pt_1[1]))#(100, 100) #x, y
    x2, y2 = (int(pt_1[0]+pt_2[1]), int(pt_1[1]+pt_2[0]))#(200, 200) #width, height

    mrk_color = line_color#(0, 100, 50)  # (B, G, R)
    # Define the thickness of the rectangle's lines
    thickness = 2

    # Calculate the center of the rectangle
    center = (x1), (y1) #((x1 + x2) // 2, (y1 + y2) // 2)

    # Calculate the size of the rectangle
    size = (x2 - x1, y2 - y1)

    # Rotation angle in degrees
    angle_degrees = angle_pt #180
    ##if angle_degrees != 0
    # Convert angle to radians
    angle_radians = np.deg2rad(angle_degrees)

    #print("CENTER",center)
    # Define the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1)

    # Define the points of the rectangle
    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)

    # Translate the rectangle so that its center is at the origin
    translated_points = points - center

    # Rotate the rectangle
    rotated_points = np.dot(rotation_matrix[:, :2], translated_points.T).T + center

    # Convert the points to integers
    rotated_points = rotated_points.astype(int)

    # Draw the rotated rectangle on the image
    # image = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.polylines(large_image_0, [rotated_points], isClosed=True, color=mrk_color, thickness=thickness)

    #cv2.rectangle(input_image_cv, pt_1, (int(pt_1[0]) + int(pt_2[1]), int(pt_1[1]) + int(pt_2[0])), line_color, thickness)


def parents_size_transform(
    current_node: dict, data_json: dict, transform_parents_acc: dict
) -> dict:
    print("parents_size_transform")
    parent_guid: dict = current_node.get("parentIndex", {}).get("guid")

    if parent_guid:
        parent_localID: float = parent_guid.get("localID")
        parent_node: dict = next(
            (
                n
                for n in data_json["nodeChanges"]
                if n["guid"]["localID"] == parent_localID
            ),
            None,
        )
        # print("transform_parents_acc", transform_parents_acc)
        if parent_node:
            parent_size = parent_node.get("size", {"x": 0, "y": 0})
            parent_transform = parent_node.get(
                "transform",
                {"m00": 0, "m01": 0, "m02": 0, "m10": 0, "m11": 0, "m12": 0},
            )
            transform_parents_acc = transform_accomulated(
                transform_parents_acc, parent_transform
            )

            # print(f"Parent localID: {parent_localID}, Size (x, y): ({parent_size['x']}, {parent_size['y']}), Transform (m02, m12): ({parent_transform['m02']}, {parent_transform['m12']})")
            return parents_size_transform(parent_node, data_json, transform_parents_acc)
    print("transform_parents_acc", transform_parents_acc)
    return transform_parents_acc

def current_node_size_transform(
    current_node: dict, data_json: dict, transform_parents: dict
) -> tuple:
    print("current_node_size_transform")
    size: dict = current_node.get("size", {"x": 0, "y": 0})
    transform: dict = current_node.get(
        "transform", {"m00": 0, "m01": 0,
                      "m02": 0, "m10": 0, "m11": 0, "m12": 0}
    )

    return size, transform

def mark_image_with_box_dict(image: Image, transform: dict, size: dict, color="red"):
    width, height = size["x"], size["y"]
    x, y = transform["m02"], transform["m12"]
    # image = Image.open(image_path)
    bounding_box = (x, y, x + width, y + height)
    outline_color = color
    draw = ImageDraw.Draw(image)
    # Draw a rectangle on the image
    draw.rectangle(bounding_box, outline=outline_color, width=2)
    return image

def crop_image_area(
    image: Image.Image, x: float, y: float, width: float, height: float
) -> Image.Image:
    # image = Image.open(image_path)
    # The crop method takes the region to crop as (left, upper, right, lower)
    cropped_image: Image.Image = image.crop((x, y, x + width, y + height))
    return cropped_image






# preprocess to eliminate blank images in "temp_match_location_finder" function
def preprocess_image(image_cv: np.ndarray) -> np.ndarray:
    gray_image_cv: np.ndarray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    # Calculate the histogram
    calc_hist = cv2.calcHist([gray_image_cv], [0], None, [256], [0, 256])
    # Check if the image is predominantly white or black
    if calc_hist[255] > 0.9 * np.prod(gray_image_cv.shape):
        # If the image is mostly white, return None
        return None
    elif calc_hist[0] > 0.9 * np.prod(gray_image_cv.shape):
        # If the image is mostly black, return None
        return None
    # If all pixels are of same color
    elif np.all(np.isclose(calc_hist, [np.prod(gray_image_cv.shape)], atol=100)):
        # If all pixels have the same color, return None
        return None
    else:
        # Otherwise, return the original image
        return image_cv
    

def temp_match_location_finder(developed_image: np.ndarray, cropped_image: Image, threshold_lvl: float):
    print("temp_match_location_finder")

    large_image = developed_image
    cropped_image_cv_01: np.ndarray = np.array(cropped_image)

    cropped_image_cv_0 = preprocess_image(cropped_image_cv_01)
    if cropped_image_cv_0 is None:
        print("Cropped image is blank or contains only a single solid color.")
        return [], None

    cropped_image_cv = cropped_image_cv_0

    # Convert cropped image to grayscale
    cropped_gray = cv2.cvtColor(cropped_image_cv, cv2.COLOR_BGR2GRAY)

    # Convert large image to grayscale
    large_gray = cv2.cvtColor(large_image, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    result = cv2.matchTemplate(large_gray, cropped_gray, cv2.TM_CCOEFF_NORMED)
    # Calculate the correlation using cross-correlation

    # Specify a threshold to find the best match
    threshold = 0.8  # threshold_lvl
    locations = np.where(result >= threshold)

    return locations, cropped_image_cv


def mark_numpy_matched(locations, rotation_angle:int, developed_image: np.ndarray, cropped_image_cv: Union[None, np.ndarray], color_mark: tuple[int, int, int], text: str):
    
    large_image = developed_image

    if 1 == 1:#np.any([arr.size != 0 for arr in locations]):
        i_len = 0
        for pt in zip(*locations[::-1]):
            print(type(pt))
            print(pt)
            print(cropped_image_cv.shape)
            rectangle_rotation(large_image, rotation_angle , pt, cropped_image_cv.shape, color_mark, 1)
            # cv2.rectangle(
            #     large_image,
            #     pt,
            #     (pt[0] + cropped_image_cv.shape[1], pt[1] + cropped_image_cv.shape[0]),
            #     (0, 255, 0), 2, )

            start_x, start_y = pt[0], pt[1]
            label_text = f"Allignment {text}"
            # Adjust vertical position
            label_position = (start_x, start_y - 10)
            cv2.putText(
                large_image,
                label_text,
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color_mark, 2,)
            i_len += 1
            if i_len == 2:
                print("long locations array pt")
                break
    return large_image

def missing_object(developed_image, x, y, width, height, rotation_angle:int, color_mark: tuple[int, int, int], text: str):
    print("missing_object")

    large_image = developed_image
    pt = (int(x), int(y))
    cv2.rectangle( large_image, pt, ( pt[0] + int(width), pt[1] + int(height) ), color_mark, 2 )

    start_x, start_y = pt[0], pt[1]
    label_text = f"Missing {text}"
    label_position = (start_x, start_y - 10)  # Adjust vertical position
    cv2.putText(
        large_image,
        label_text,
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, color_mark, 1,
    )
    return large_image

def moved_object(locations: tuple[np.ndarray, ...], large_image):
    print("moved_object")

def save_image(image, save_path):
    image.save(save_path)

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def process_images_json_design(design_image_path:str, developer_image_path:str):
    print("process_images_json_design")

    output_directory:str = '/Users/home/Desktop/GSK/Computer-Vision/demo-app-web/processed'
    print( '18',output_directory)
    if not os.path.exists(output_directory):
        print("not os.path.exists(output_directory)")
        os.makedirs(output_directory)

    json_path = ("/Users/home/Desktop/GSK/Computer-Vision/yolo-test/custom-data/json-design.json")
    # design_image_path = "/Users/home/Desktop/GSK/Computer-Vision/yolo-test/custom-data/Power AgencyWeb Design.png"
    # developed_image_path = "/Users/home/Desktop/GSK/Computer-Vision/yolo-test/custom-data/PowerAgencyWeb_Developed.png"
    
    #design_image_path = design_image_path
    developed_image_path = developer_image_path

    with open(f"{json_path}") as f:
        data_json: dict = json.load(f)

    image_des: Image = Image.open(design_image_path)
    image_dev: Image = Image.open(developed_image_path)
    # numpy array dev image
    numpy_array_image_d = cv2.imread(f"{design_image_path}")
    numpy_array_image_dev = cv2.imread(f"{developed_image_path}")

    design_image_name:str = os.path.basename(design_image_path)
    developer_image_name:str = os.path.basename(developed_image_path)
    processed_path_1 = os.path.join(output_directory, 'processed_' + design_image_name)
    processed_path_2 = os.path.join(output_directory, 'processed_' + developer_image_name)


    
    color_def: tuple[int, int, int] = (0, 0, 0)
    color_red: tuple[int, int, int] = (255, 0, 0)
    color_green: tuple[int, int, int] = (0, 255, 0)
    color_blue: tuple[int, int, int] = (0, 0, 255)
    marker_colors = [color_def, color_red, color_green, color_blue]

    node_mapping: dict = {node['guid']['localID']: node for node in data_json['nodeChanges']}
    # print("", node_mapping)
    transform: dict = {'m00': 0, 'm01': 0, 'm02': 0, 'm10': 0, 'm11': 0, 'm12': 0}
    
    large_image_cv = numpy_array_image_dev
    marker_color: tuple[int, int, int] = (0, 200, 0)


    #PROCESS
    for node in data_json['nodeChanges']: #[96:140]:
        # print(node)
        print("#####node['guid']", node['guid']['localID'])
        # == "FILL":
        if 'styleType' in node and node['styleType'] in ["FILL", "EFFECT"]:
            print("----node['guid']", node['guid']['localID'])
        else:
            # if cropped_image.size != (0, 0):
            #     print("cropped_image.size")
            # get_parent_info(node, data_json, transform)
            # print("---",get_parent_info(node, data_json))
            parents_transform: dict = parents_size_transform(
                node, data_json, transform)
            print(node['guid']['localID'], "?????????", parents_transform)
            node_size, node_transform = current_node_size_transform(
                node, data_json, parents_transform)
            node_acc_transform = transform_accomulated(
                node_transform, parents_transform)

            width, height = node_size['x'], node_size['y']
            x, y = node_acc_transform['m02'], node_acc_transform['m12']
            if all(node_size[key] != 0 for key in ['x', 'y']):  # initial chek if object exist

                croped_image = crop_image_area(image_des, x, y, width, height)
                #display(croped_image)
                location, crope = temp_match_location_finder(
                    large_image_cv, croped_image, 0.9)

                node_id = str(node['guid']['localID'])
                rotation_angle = int(rotation_factor(node_acc_transform)) #Degree is in float
                print(node_id, "-------", location)
                if location != []:
                    print("location is blank color []")
                    if np.any([arr.size != 0 for arr in location]):
                        if x not in location[1] and y not in location[0]:
                            large_image_cv = mark_numpy_matched(
                                location, rotation_angle, large_image_cv, crope, marker_colors[2], str(node_id))

                    if np.any([arr.size == 0 for arr in location]):
                        large_image_cv = missing_object(
                            large_image_cv, x, y, width, height, rotation_angle, marker_colors[1], str(node_id))

                    if x not in location[1] and y not in location[0]:
                        print("test", location[0])



    # End save
    
    img1_processed = numpy_array_image_d
    img2_processed = large_image_cv
    print( '77',processed_path_2)

    cv2.imwrite(processed_path_1, img1_processed)
    cv2.imwrite(processed_path_2, large_image_cv)

    print(f"Processed images saved successfully to: {processed_path_1} and {processed_path_2}")


###################################################################
# des = "/Users/home/Desktop/GSK/Computer-Vision/demo-app-web/uploads/image_1.jpg"
# dev = "/Users/home/Desktop/GSK/Computer-Vision/demo-app-web/uploads/image_2.jpg"
# process_images_json_design(des, dev)


# Processed images saved successfully to: /Users/home/Desktop/GSK/Computer-Vision/demo-app/processed/f_processed_image_1.jpg and /Users/home/Desktop/GSK/Computer-Vision/demo-app/processed/f_processed_image_2.jpg





"""
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

"""