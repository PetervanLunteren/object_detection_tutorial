# Peter van Lunteren, 6 march 2022.
# Part of the object detection tutorial https://github.com/PetervanLunteren/object_detection_tutorial

# This script changes the folderstucture of the directory 'object_detection'. It creates the 'training',
# 'images', and 'exported_model' dirs and removes the content of the 'data' dir. It als copies the python
# files from the 'object_detection_tutorial' dir into the 'object_detection' dir. It will ensure that the
# tutorial will run smoothly.

# import packages
import os
import shutil
from pathlib import Path

# create dirs
object_detection = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

shutil.rmtree(os.path.join(object_detection, "data"))
Path(os.path.join(object_detection, 'data')).mkdir(parents=True, exist_ok=True)
print("\nContents of folder 'data' removed :", os.path.join(object_detection, "data"))

Path(os.path.join(object_detection, 'training')).mkdir(parents=True, exist_ok=True)
print("\nFolder 'training' created :", os.path.join(object_detection, 'training'))

Path(os.path.join(object_detection, 'images')).mkdir(parents=True, exist_ok=True)
print("\nFolder 'images' created :", os.path.join(object_detection, 'images'))

Path(os.path.join(object_detection, 'exported_model')).mkdir(parents=True, exist_ok=True)
print("\nFolder 'exported_model' created :", os.path.join(object_detection, 'exported_model'), "\n")

# remove py files
current_dir = os.path.basename(os.path.normpath(os.path.dirname(os.path.abspath(__file__))))
if current_dir == "object_detection_tutorial":
    src = os.path.dirname(os.path.abspath(__file__))
    dest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = os.listdir(src)
    for f in files:
        if f.endswith('.py') and f != "change_folder_structure.py":
            shutil.move(os.path.join(src, f), dest)