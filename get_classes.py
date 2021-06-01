import pandas as pd
import os

test_labels = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "test_labels.csv"))
train_labels = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "train_labels.csv"))

all_classes = list(test_labels['class']) + list(train_labels['class'])
all_classes = list(dict.fromkeys(all_classes))

string1 = "def class_text_to_int(row_label):\n"
string2 = ""
for class_ in all_classes:
    string1 += "\tif row_label == '" + class_ + "':\n\t\treturn " + str(all_classes.index(class_) + 1) + "\n"
    string2 += "item {\n\tid: " + str(all_classes.index(class_) + 1) + "\n\tname: '" + class_ + "'\n}\n"

print(string1 + "\n\n")
print(".pbtxt file: \n", string2)