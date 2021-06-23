import numpy as np
import os
import shutil
from pathlib import Path
import sys

folder_to_be_separated = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
prop_to_test = float(sys.argv[1])

all_files_w_ext = [f for f in os.listdir(folder_to_be_separated)
                   if os.path.isfile(os.path.join(folder_to_be_separated, f)) and not f.endswith(".DS_Store")]
all_files_wt_ext = [os.path.splitext(f)[0] for f in os.listdir(folder_to_be_separated) if
                    os.path.isfile(os.path.join(folder_to_be_separated, f)) and not f.endswith(".DS_Store")]
all_files_wt_ext = list(dict.fromkeys(all_files_wt_ext))

random_files = np.random.choice(all_files_wt_ext, int(len(all_files_wt_ext) * prop_to_test), replace=False)

Path(os.path.join(folder_to_be_separated, 'train')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(folder_to_be_separated, 'test')).mkdir(parents=True, exist_ok=True)

for file_base in random_files:
    for file in all_files_w_ext:
        if file.startswith(file_base):
            print(file, "moved to test")
            shutil.move(os.path.join(folder_to_be_separated, file),
                        os.path.join(folder_to_be_separated, 'test'))

for file in os.listdir(folder_to_be_separated):
    if os.path.isfile(os.path.join(folder_to_be_separated, file)) and not file.endswith(".DS_Store"):
        print(file, "moved to train")
        shutil.move(os.path.join(folder_to_be_separated, file),
                    os.path.join(folder_to_be_separated, 'train'))