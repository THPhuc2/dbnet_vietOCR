import os 
import shutil
import cv2
import numpy as np

def convert_str_to_yolo(string):
    class_id, x_center, y_center, width, height = list(map(float, string.split(" ")))
    
    return [x_center, y_center, width, height]

def convert_yolo_to_bbox(yolo_format):
    x_center, y_center, width, height = yolo_format
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    
    return [x_min, y_min, x_max, y_max]

def convert_bbox_to_dbnet_bbox(bbox, width, height):
    x_min, y_min, x_max, y_max = bbox 
    
    return [
        int(x_min * width), 
        int(y_min * height), 
        int(x_max * width), 
        int(y_max * height)
    ]

def convert_str_to_dbnet_bbox(string, height, width):
    yolo_format = convert_str_to_yolo(string)
    bbox = convert_yolo_to_bbox(yolo_format)
    bbox = convert_bbox_to_dbnet_bbox(bbox, width, height)
    
    return bbox

def convert_bbox_to_str(bbox):
    x_min, y_min, x_max, y_max = bbox 
    return "{},{},{},{},{},{},{},{}".format(x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max)

def extend_bbox(bbox, ratio_height=0.0, ratio_width=0.0):
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin

    xmin -= ratio_width * width
    xmax += ratio_width * width
    ymin -= ratio_height * height
    ymax += ratio_height * height

    return [convert_int(xmin), convert_int(ymin), convert_int(xmax), convert_int(ymax)]

def convert_int(x):
    return int(x) if x >= 0 else 0 

def convert_yolo_file_to_dbnet_format(file_path, new_file_path, height, width):
    assert os.path.exists(file_path), "This path is not exists"
    result_file = open(new_file_path, "w")
    
    with open(file_path, "r") as f:
        for line in f.readlines():
            # Delete \n
            line = line[:-1]
            # Get dbnet bbox 
            bbox = convert_str_to_dbnet_bbox(line, height, width)
            bbox = extend_bbox(bbox)
            string = convert_bbox_to_str(bbox)
            
            result_file.write(string + "," + "\n")
            
    result_file.close()
