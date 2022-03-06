# This script takes a tensorflow 2 trained object detection model and uses it on a directory of images. Output is
# csv file with all detections (if specified), images with detections drawn on the image (if specified) and print
# statements in the terminal.

# Peter van Lunteren, 6 march 2022.
# Part of the object detection tutorial https://github.com/PetervanLunteren/object_detection_tutorial

# import required packages
import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import csv
import argparse

from utils import label_map_util
from utils import visualization_utils as vis_util
from detector import DetectorTF2

# get user input
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image_directory", help="Path to directory with images which you want to process",
                    required=True, type=str)
parser.add_argument("-t", "--threshold",
                    help="Detections above this threshold will be processed. Must be value between0 and 1. "
                         "Default is 0.7",
                    default=0.7, type=float)
parser.add_argument('--no_draw_box', dest='draw_box', action='store_false',
                    help="Specify if you do not want the detections to be visualised in output")
parser.set_defaults(draw_box=True)
parser.add_argument('--no_export_csv', dest='export_csv', action='store_false',
                    help="Specify if you do not want a csv file with detections")
parser.set_defaults(export_csv=True)

args = parser.parse_args()
image_directory = args.image_directory
threshold = args.threshold
draw_box = args.draw_box
export_csv = args.export_csv

# retrieve the paths to the frozen model and label file
PATH_TO_OBJDET = os.path.dirname(os.path.abspath(__file__))
PATH_TO_PB = os.path.join(PATH_TO_OBJDET, "exported_model", "saved_model")
PATH_TO_LABELS = os.path.join(PATH_TO_OBJDET, "data", "object_detection.pbtxt")

# open model
detector = DetectorTF2(PATH_TO_PB, PATH_TO_LABELS, class_id=None)

# initiate header of csv file
csv_content = [["image_path", "class", "confidence", "ymin", "xmin", "ymax", "xmax", "image_height", "image_width"]]

# process all images in dir
for filename in os.listdir(image_directory):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
        PATH_TO_IMG = os.path.join(image_directory, filename)
        frame = cv2.imread(PATH_TO_IMG)
        height, width, _ = frame.shape
        detections = detector.DetectFromImage(frame)
        n_detections_above_thresh = len([element[5] for element in detections if element[5] > threshold])

        # remove detections below the threshold
        detections = [detection for detection in detections if detection[5] >= threshold]

        # print detections in terminal
        print(f"\nImage {PATH_TO_IMG} has {int(n_detections_above_thresh)} detection(s):")
        i = 0
        for detection in detections:
            conf = float(detection[5])
            detection_class = detection[4]
            ymin = detection[0]
            xmin = detection[1]
            ymax = detection[2]
            xmax = detection[3]

            print(f"   Detection {i+1}:")
            print(f"      class        = {detection_class}")
            print(f"      confidence   = {round(conf, 3)}")
            print(f"      ymin         = {ymin}")
            print(f"      xmin         = {xmin}")
            print(f"      ymax         = {ymax}")
            print(f"      xmax         = {xmax}")
            print(f"      image height = {height}")
            print(f"      image width  = {width}\n")

            # fill list with csv content
            if export_csv:
                csv_content.append([PATH_TO_IMG, detection_class, conf, ymin, xmin, ymax, xmax, height, width])

        # draw boxes around the detections
        if draw_box:
            img_box = detector.DisplayDetections(frame, detections)
            Path(os.path.join(image_directory, "_boxes")).mkdir(parents=True, exist_ok=True)
            OUT_IMG = os.path.join(image_directory, "_boxes", "box_" + filename)
            cv2.imwrite(OUT_IMG, img_box)

# export csv file
if export_csv:
    csv_file = os.path.join(image_directory, "_results.csv")
    csv_file = open(csv_file, 'w')
    csv_writer = csv.writer(csv_file, delimiter=",")
    for row in csv_content:
        csv_writer.writerow(row)
    csv_file.close()
