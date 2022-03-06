# Peter van Lunteren, 6 march 2022.
# Part of the object detection tutorial https://github.com/PetervanLunteren/object_detection_tutorial

# This script will move a proportion of random images and associated xml files in the 'test' and 'train'
# dir. The proportion which will go to the 'test' dir can be adjusted with the --prop_to_test argument.

# import packages
import numpy as np
import os
import shutil
from pathlib import Path
import sys
import argparse

# get user input
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prop_to_test",
                    help="Sets the proportion of images which will be randomly chosen en moved to the "
                    "'test' dir. The remaining images will be moved to the 'train' dir. Default is 0.2",
                    default=0.2, type=float)
args = parser.parse_args()
prop_to_test = args.prop_to_test

# set path
folder_to_be_separated = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')

# randomly choose images
all_files_w_ext = [f for f in os.listdir(folder_to_be_separated)
                   if os.path.isfile(os.path.join(folder_to_be_separated, f)) and not f.endswith(".DS_Store")]
all_files_wt_ext = [os.path.splitext(f)[0] for f in os.listdir(folder_to_be_separated) if
                    os.path.isfile(os.path.join(folder_to_be_separated, f)) and not f.endswith(".DS_Store")]
all_files_wt_ext = list(dict.fromkeys(all_files_wt_ext))

random_files = np.random.choice(all_files_wt_ext, int(len(all_files_wt_ext) * prop_to_test), replace=False)

# create dirs
Path(os.path.join(folder_to_be_separated, 'train')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(folder_to_be_separated, 'test')).mkdir(parents=True, exist_ok=True)

# move random images and associated xml files to 'test'
for file_base in random_files:
    for file in all_files_w_ext:
        if os.path.splitext(file)[0] == file_base:
            shutil.move(os.path.join(folder_to_be_separated, file),
                        os.path.join(folder_to_be_separated, 'test'))
            print(file, "moved to test")

# move the remaining part to 'train'
for file in os.listdir(folder_to_be_separated):
    if os.path.isfile(os.path.join(folder_to_be_separated, file)) and not file.endswith(".DS_Store"):
        shutil.move(os.path.join(folder_to_be_separated, file),
                    os.path.join(folder_to_be_separated, 'train'))
        print(file, "moved to train")