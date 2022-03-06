# Machine learning for dummies - train and use your own object detection model in a few easy steps 

### Step 0: Prerequisites

### Step 1: Installation (just once)
```batch
conda create -n ObjectDetectionTutorial_TF2 python==3.7 -y
conda activate ObjectDetectionTutorial_TF2
pip install tensorflow==2.8.0
conda install -c anaconda protobuf -y
PATH_TO_CONDA_DIR=`conda info | grep 'base environment' | cut -d ':' -f 2 | xargs | cut -d ' ' -f 1`
cd "$PATH_TO_CONDA_DIR/envs/ObjectDetectionTutorial_TF2/lib/python3.7/site-packages/tensorflow"
git clone https://github.com/tensorflow/models.git
cd models
git checkout c9ae0d833800f90d828a51f0be47ac4a083165bc
cd research/object_detection
git clone https://github.com/PetervanLunteren/object_detection_tutorial
pip install -r object_detection_tutorial/requirements_TF2.txt
```

### Step 2: Start-up (every time you start again)
```batch
conda activate ObjectDetectionTutorial_TF2
PATH_TO_CONDA_DIR=`conda info | grep 'base environment' | cut -d ':' -f 2 | xargs | cut -d ' ' -f 1`
cd "$PATH_TO_CONDA_DIR/envs/ObjectDetectionTutorial_TF2/lib/python3.7/site-packages/tensorflow/models/research"
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd object_detection
python builders/model_builder_test.py
```

### Step 3: Change folder structure
```batch
cd object_detection_tutorial
python change_folder_structure.py
cd ..
```

### Step 4: Copy images

### Step 5: Label images
```batch
cd ..
git clone https://github.com/tzutalin/labelImg
cd labelImg
git checkout 276f40f5e5bbf11e84cfa7844e0a6824caf93e11
pyrcc5 -o libs/resources.py resources.qrc
PATH_TO_IMG_DIR=$PATH_TO_CONDA_DIR/envs/ObjectDetectionTutorial_TF2/lib/python3.7/site-packages/tensorflow/models/research/object_detection/images
PATH_TO_CLASSES=$PATH_TO_CONDA_DIR/envs/ObjectDetectionTutorial_TF2/lib/python3.7/site-packages/tensorflow/models/research/labelImg/data/predefined_classes.txt
python labelImg.py "$PATH_TO_IMG_DIR" "$PATH_TO_CLASSES" "$PATH_TO_IMG_DIR"
cd ..
cd object_detection
```

### Step 6: Separate test and train images
```batch
python move_random_files.py --prop_to_test 0.2
```

### Step 7: Create required files
```batch
python xml_to_csv.py
python get_classes.py
python generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record --image_dir=images/train/
python generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record --image_dir=images/test/
```

### Step 8: Download model


### Step 9: Adjust .config file


### Step 10: Train
```batch
pmset noidle &
PMSETPID=$!
python3 model_main_tf2.py --train_dir=training/ --pipeline_config_path=data/pipeline.config --model_dir=training/ –logtostderr
kill $PMSETPID
```

### Step 11: Evaluate
```batch
tensorboard --logdir training/
```

### Step 12: Export model
```batch
python exporter_main_v2.py --input_type image_tensor --pipeline_config_path data/pipeline.config --trained_checkpoint_dir training/ --output_directory exported_model/
```

### Step 13: Use model
```batch
python use_model_TF2.py --image_directory "$PATH_TO_CONDA_DIR/envs/ObjectDetectionTutorial_TF2/lib/python3.7/site-packages/tensorflow/models/research/object_detection/object_detection_tutorial/new_images_to_test" --threshold 0.8
```
