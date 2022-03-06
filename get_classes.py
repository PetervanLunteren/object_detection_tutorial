# Peter van Lunteren, 6 march 2022.
# Part of the object detection tutorial https://github.com/PetervanLunteren/object_detection_tutorial

# This script reads the 'test_labels.csv' and 'train_labels.csv' and writes the 'object_detection.pbtxt'.

# import packages
import pandas as pd
import os

# define function to write pbtxt file
def write_pbtxt(path, content):
    with open(str(path), 'a') as the_file:
        the_file.write(content)
    print("pbtxt file writen to :", path, "\n")


# set paths
test_labels = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "test_labels.csv"))
train_labels = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "train_labels.csv"))
path_pbtxt_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "object_detection.pbtxt")

# read unique classes and sort them alphabetically
all_classes = sorted(list(dict.fromkeys(list(test_labels['class']) + list(train_labels['class']))))

# write pbtxt file
str_pbtxt = ""
for class_ in all_classes:
    str_pbtxt += "item {\n  id: " + str(all_classes.index(class_) + 1) + "\n  name: '" + class_ + "'\n}\n"
write_pbtxt(path_pbtxt_data, str_pbtxt)
