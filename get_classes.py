import pandas as pd
import os

def write_pbtxt(path, content):
    with open(str(path), 'a') as the_file:
        the_file.write(content)
    print("pbtxt file writen to :", path)

test_labels = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "test_labels.csv"))
train_labels = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "train_labels.csv"))
path_pbtxt_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "object_detection.pbtxt")
path_pbtxt_training = os.path.join(os.path.dirname(os.path.realpath(__file__)), "training", "object_detection.pbtxt")

all_classes = list(test_labels['class']) + list(train_labels['class'])
all_classes = list(dict.fromkeys(all_classes))

str_def = "def class_text_to_int(row_label):\n"
str_pbtxt = ""
for class_ in all_classes:
    str_def += "   if row_label == '" + class_ + "':\n      return " + str(all_classes.index(class_) + 1) + "\n"
    str_pbtxt += "item {\n\tid: " + str(all_classes.index(class_) + 1) + "\n\tname: '" + class_ + "'\n}\n"

print(str_def + "\n\n")

write_pbtxt(path_pbtxt_data, str_pbtxt)
write_pbtxt(path_pbtxt_training, str_pbtxt)