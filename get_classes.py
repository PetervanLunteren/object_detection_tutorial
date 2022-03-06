# Peter van Lunteren, 6 march 2022.
# Part of the object detection tutorial https://github.com/PetervanLunteren/object_detection_tutorial

# This script reads the 'test_labels.csv' and 'train_labels.csv' and writes the 'object_detection.pbtxt'.
# It also prints out the function which should be adjusted in 'generate_tfrecords.py'.

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

# read unique classes
all_classes = list(test_labels['class']) + list(train_labels['class'])
all_classes = list(dict.fromkeys(all_classes))
all_classes = sorted(all_classes)

# print to terminal and pbtxt file
str_def = "def class_text_to_int(row_label):\n"
str_pbtxt = ""
for class_ in all_classes:
    str_def += "    if row_label == '" + class_ + "':\n        return " + str(all_classes.index(class_) + 1) + "\n"
    str_pbtxt += "item {\n  id: " + str(all_classes.index(class_) + 1) + "\n  name: '" + class_ + "'\n}\n"
print("Part to be copy pasted into generate_tf_record.py:\n\n" + str_def)
write_pbtxt(path_pbtxt_data, str_pbtxt)
