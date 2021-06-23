import os
import shutil
from pathlib import Path

object_detection = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.rmdir(os.path.join(object_detection, "data"))
Path(os.path.join(object_detection, 'training')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(object_detection, 'images')).mkdir(parents=True, exist_ok=True)

current_dir = os.path.basename(os.path.normpath(os.path.dirname(os.path.abspath(__file__))))
if current_dir == "object_detection_files":
    src = os.path.dirname(os.path.abspath(__file__))
    dest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = os.listdir(src)
    for f in files:
        if f.endswith('.py') and f != "change_folder_structure.py":
            shutil.move(os.path.join(src, f), dest)