# Machine learning for dummies - train and use your own object detection model in a few easy steps

**THIS TUTORIAL IS WORK IN PROGRESS**


This is a tutorial which makes it easy for everybody to train their own custom model to recogise objects in images. When I started with object detection models, I tried many toturials which generally ended up with some kind of error. I try to avoid that with this tutorial by automating as many steps as possible and working with an anaconda environment. For each step I will first explain what you need to do, then show you the commands to actually do it. For the sake of explanation I will train a model to detect the number of eyes on dice roll, but you can use this tutorial to train whatever you want with whatever model you want.

## Step 0: Prerequisites
First of all you'll need to install [Anaconda](https://www.anaconda.com/products/individual). Need help? Follow [these steps](https://docs.anaconda.com/anaconda/install/mac-os/). Anaconda allows you to create environments in which you can install certain versions of packages, so that there won't be any version errors along the way.

TO DO: add conda to path, or activate it every time you start
```
conda -V
```
if output is something like `zsh: command not found: conda`, you'll have to tell your computer where to find anaconda. 

```
source /users/m1/anaconda3/bin/activate
```
Then try it:
```
conda -V
```

## Step 1: Installation (just once)
In this step we'll install all neccesary packages and repositories. You only have to complete this step once. First, I'll create an environment in which we'll work during this tutorial. Open a new window in the Terminal application and enter the following commands.
```batch
conda create -n ObjectDetectionTutorial_TF2 python==3.7 -y
conda activate ObjectDetectionTutorial_TF2
```
Then I install tensorflow and protobuf inside this environment.
```batch
pip install tensorflow==2.9.2
conda install -c anaconda protobuf=3.13.0.1 -y
```
The following command finds the path to your anaconda directory - which we will change directory into to download repositories. 
```batch
PATH_TO_CONDA_DIR=`conda info | grep 'base environment' | cut -d ':' -f 2 | xargs | cut -d ' ' -f 1`
cd "$PATH_TO_CONDA_DIR/envs/ObjectDetectionTutorial_TF2/lib/python3.7/site-packages/tensorflow"
```
Download the `models` repo from Tensorflow and my own repo with scripts and files we'll need further on. As you can see I `checkout` the `models` repo. That means I use a specific version, because I don't know what is going to happen with that repo after I publish this tutorial. I know that this combination works.
```batch
git clone https://github.com/tensorflow/models.git
cd models
git checkout c9ae0d833800f90d828a51f0be47ac4a083165bc
cd research/object_detection
git clone https://github.com/PetervanLunteren/object_detection_tutorial
```
Now we only have to install the rest of the packages using the `requirements_TF2.txt` file and exit the window.
```batch
pip install -r object_detection_tutorial/requirements_TF2.txt
exit
```
You can now close the terminal window. For the next steps we'll use a new window.

```
function fetch_os_type() {
  echo "Checking OS type and version..." 2>&1 | tee -a "$LOG_FILE"
  OSver="unknown"  # default value
  uname_output="$(uname -a)"
  echo "$uname_output" 2>&1 | tee -a "$LOG_FILE"
  # macOS
  if echo "$uname_output" | grep -i darwin >/dev/null 2>&1; then
    # Fetch macOS version
    sw_vers_output="$(sw_vers | grep -e ProductVersion)"
    echo "$sw_vers_output" 2>&1 | tee -a "$LOG_FILE"
    OSver="$(echo "$sw_vers_output" | cut -c 17-)"
    macOSmajor="$(echo "$OSver" | cut -f 1 -d '.')"
    macOSminor="$(echo "$OSver" | cut -f 2 -d '.')"
    # Make sure OSver is supported
    if [[ "${macOSmajor}" = 10 ]] && [[ "${macOSminor}" < "${MACOSSUPPORTED}" ]]; then
      die "Sorry, this version of macOS (10.$macOSminor) is not supported. The minimum version is 10.$MACOSSUPPORTED." 2>&1 | tee -a "$LOG_FILE"
    fi
    # Fix for non-English Unicode systems on MAC
    if [[ -z "${LC_ALL:-}" ]]; then
      export LC_ALL=en_US.UTF-8
    fi

    if [[ -z "${LANG:-}" ]]; then
      export LANG=en_US.UTF-8
    fi
    OS="osx"
  # Linux
  elif echo "$uname_output" | grep -i linux >/dev/null 2>&1; then
    OS="linux"
  else
    die "Sorry, the installer only supports Linux and macOS, quitting installer" 2>&1 | tee -a "$LOG_FILE"
  fi
}
fetch_os_type

case $OS in
linux*)
  if ! lscpu | grep -q " avx "; then
    echo "AVX is not supported by this CPU! Installing a version of TensorFlow compiled for older CPUs..." 2>&1 | tee -a "$LOG_FILE"
    pip uninstall tensorflow -y
    pip install https://github.com/spinalcordtoolbox/docker-tensorflow-builder/releases/download/v1.15.5-py3.7/tensorflow-1.15.5-cp37-cp37m-linux_x86_64.whl
  else
    echo "AVX is supported on this Linux machine. Keeping default TensorFlow..." 2>&1 | tee -a "$LOG_FILE"
  fi
  ;;
osx)
  if sysctl machdep.cpu.brand_string | grep -q "Apple M1"; then
    echo "AVX is not supported on M1 Macs! Installing a version of TensorFlow compiled for M1 Macs..." 2>&1 | tee -a "$LOG_FILE"
    pip uninstall tensorflow -y
    pip install https://github.com/alessandro893/tensorflow-macos-no_avx/releases/download/v2.9.2/tensorflow-2.9.2-cp38-cp38-macosx_12_0_x86_64.whl
    pip freeze
  else
    echo "AVX is supported on this macOS machine. Keeping default TensorFlow..." 2>&1 | tee -a "$LOG_FILE"
  fi
  ;;
esac
```


## Step 2: Start-up (every time you start again)
Now that the installation is completed, you only have to start up your environment every time you want to access it. Here we activate and change directory.
```batch
conda activate ObjectDetectionTutorial_TF2
PATH_TO_CONDA_DIR=`conda info | grep 'base environment' | cut -d ':' -f 2 | xargs | cut -d ' ' -f 1`
cd "$PATH_TO_CONDA_DIR/envs/ObjectDetectionTutorial_TF2/lib/python3.7/site-packages/tensorflow/models/research"
```
Execute the protobuf compile and set `PYTHONPATH`.
```batch
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
Change directory and test if everything is set.
```batch
cd object_detection
python builders/model_builder_test.py
```
If the there is no error message, you're ready to go.

## Step 3: Change folder structure
Here we execute a python script which creates the directories 'training', 'images', and 'exported_model'. It also places all the other python scripts I've written for this tutorial in the 'object_detection' directory.
```batch
cd object_detection_tutorial
python change_folder_structure.py
cd ..
```

## Step 4: Copy images
Now you can place the images you want to use for the training in the `images` directory. I've prepared some images of dice-rolls if you don't have images yourself and just want to follow along with this tutorial. You can find them in `object_detection_tutorial/images_and_labels/images`. 

After all these steps your folder will look like this:
```
object_detection
	|--- data
	|--- training
	|--- images
	|	|--- image1.jpg
	|	|--- image2.jpg
	|	|--- image3.jpg
	|	|--- …
	|--- …
```

## Step 5: Label images
For training it is important that the computer knows what to find on the images, and where they can find it. That's why we accompany every image with an xml file. Labeling your images is definately the most boring part of creating your own object detection model. You can use the handy <a href="https://github.com/tzutalin/labelImg">labelImg</a> program for this.

If your using the dice-roll dataset, I've already labelled it for you: `object_detection_tutorial/images_and_labels/labels`. Copy-paste this to the `images` directory.

If you want to use the labelImg program, execute the following commands. Here we clone the labelImg repo, compile it, and open it with the paths automatically set.
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
After all these steps your folder will look like this:
```
object_detection
	|--- data
	|--- training
	|--- images
	|	|--- image1.jpg
	|	|--- image1.xml
	|	|--- image2.jpg
	|	|--- image2.xml
	|	|--- image3.jpg
	|	|--- image3.xml
	|	|--- …
	|--- …
```

### Step 6: Separate test and train images
We need to seperate the images in a train and test set. We'll use 90% of the images for training and 10% for testing. The following command executes a python file which will create the directories `train` and `test` and randomly choose 10% of the images and its corresponding xml files to move to `test` (and the rest to `train`). If you want a different proportion, just change the `--proportion_to_test` argument. 
```batch
python move_random_files.py --prop_to_test=0.2
```

## Step 7: Create required files
In this step we'll generate some files required for training. First we'll use all the label files to create a `.csv` file for train and test. Then we'll execute `get_classes.py` which will read these `.csv`'s and write a `.pbtxt` file inside the `data` folder. A `.pbtxt` file is nothing more than a text file stating how the computer can convert the classes to integers. Lastly, we'll create TFRecords (tensorflow's own binary storage format) for the train- and testset.  
```batch
python xml_to_csv.py
python get_classes.py
python generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record --image_dir=images/train/
python generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record --image_dir=images/test/
```

## Step 8: Download model
Here in this step you will choose a model from the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). Here you can pick one that is suitable for your data and purpose. There are accurate models which are slow and complex (like the Faster R-CNN Inception models), but also quick and light models like the SSD MobileNet. Have a look - it's like a menu. Since recognising the number of black dots on a white dice is not too complicated, for this tutorial we'll use the `SSD MobileNet V2 FPNLite 320x320` model. If you're training something more complicated, you can choose a more complex model. Download the desired model and untar the `.tar` file. It will unpack to the folder which I will call the model folder for the rest of this tutorial. Each model will result in a different model folder. For example, if you downloaded the `SSD MobileNet V2 FPNLite 320x320` model, the model folder will be `ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8`. Now place this model folder in the `object_detection` directory.

## Step 9: Adjust .config file
In this step we will adjust the `.config` file with the appropriate parameter settings and place it in the correct location. Please note that if you're using the `ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8` model and are training on the dice-roll dataset, there is already a pre-filled `.config` file on the correct location, so you can simply skip this step. For those who are training their custom model: open the `pipeline.config` file inside the model folder and adjust the following parameters.

Number of classes. You should change the `num_classes` to the number of classes you are training for. In my case there are 6 classes: 'one', 'two', 'tree', 'four', 'five' and 'six' dots.
```
num_classes: 6
```

Batch size. Using a larger `batch_size` decreases the quality of the model, as measured by its ability to generalize. However, a `batch_size` too small has the risk of making learning too stochastic. Either way, the higher the batch size, the more memory space you’ll need so it depends on your local machine what you can handle. For this tutorial I choose a `batch_size` of 4 because you then won't need a powerfull computer to run it.
```
batch_size: 4
```

Fine tune checkpoint. Specify the `fine_tune_checkpoint` as `"<model_folder>/checkpoint/ckpt-0"`. For example:
```
fine_tune_checkpoint: "ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/ckpt-0"
fine_tune_checkpoint: "ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/ckpt-1"
```

Fine tune checkpoint type. Change the `fine_tune_checkpoint_type` to `"detection"`.
```
fine_tune_checkpoint_type: "detection"
```

Train input reader paths. Use the below `label_map_path` and `input_path` for the `train_input_reader`.
```
label_map_path: "data/object_detection.pbtxt"
input_path: "data/train.record"
```

Eval input reader paths. Use the below `label_map_path` and `input_path` for the `eval_input_reader`.
```
label_map_path: "data/object_detection.pbtxt"
input_path: "data/test.record"
```

The rest of the parameters are good to go - you don't need to change them. Now copy and place this `pipeline.config` file to your `data` directory. My pre-filled one is also there, so you can replace it. 

## Step 10: Train
It's finally time to train! You can do that with the following command. Here we'll train for 200.000 steps, but you're free to change the `--num_train_steps` to whatever you want. Please note that the `pmset` command disables your computer to sleep and will be automatically lifted after the training is completed. However, for this it's important that you execute these 4 commands in one go.  
```batch
pmset noidle &
PMSETPID=$!
python3 model_main_tf2.py --train_dir=training/ --pipeline_config_path=data/pipeline.config --model_dir=training/ –logtostderr
kill $PMSETPID
```

```batch
pmset noidle &
PMSETPID=$!
python model_main_tf2.py --pipeline_config_path=data/pipeline.config --model_dir=training/ --alsologtostderr
kill $PMSETPID
```

```
pmset noidle &
PMSETPID=$!
python3 model_main_tf2.py \
    --pipeline_config_path=data/pipeline.config \
    --model_dir=training/ \
    --alsologtostderr \
    --num_train_steps=20000 \
    --sample_1_of_n_eval_examples=1 \
    --num_eval_steps=1000
kill $PMSETPID
```

It'll print many lines - most of which you can ignore. You'll probably see `Use fn_output_signature instead` for a while. Please be patient. This can hold for a few minutes. If you see somthing like below the training has started!
```
INFO:tensorflow:{'Loss/classification_loss': 0.80173784,
 'Loss/localization_loss': 0.62153286,
 'Loss/regularization_loss': 0.15319578,
 'Loss/total_loss': 1.5764666,
 'learning_rate': 0.0319994}
I0316 17:16:21.804826 4532760000 model_lib_v2.py:708] {'Loss/classification_loss': 0.80173784,
 'Loss/localization_loss': 0.62153286,
 'Loss/regularization_loss': 0.15319578,
 'Loss/total_loss': 1.5764666,
 'learning_rate': 0.0319994}
INFO:tensorflow:Step 200 per-step time 3.216s
I0316 17:21:43.447596 4532760000 model_lib_v2.py:707] Step 200 per-step time 3.216s
INFO:tensorflow:{'Loss/classification_loss': 0.53597915,
 'Loss/localization_loss': 0.42327553,
 'Loss/regularization_loss': 0.15312059,
 'Loss/total_loss': 1.1123753,
 'learning_rate': 0.0373328}
I0316 17:21:43.447819 4532760000 model_lib_v2.py:708] {'Loss/classification_loss': 0.53597915,
 'Loss/localization_loss': 0.42327553,
 'Loss/regularization_loss': 0.15312059,
 'Loss/total_loss': 1.1123753,
 'learning_rate': 0.0373328}
```


## Step 11: Evaluate
```batch
tensorboard --logdir training/
```

## Step 12: Export model
```batch
python exporter_main_v2.py --input_type image_tensor --pipeline_config_path data/pipeline.config --trained_checkpoint_dir training/ --output_directory exported_model/
```

## Step 13: Use model
```batch
python use_model_TF2.py --image_directory "$PATH_TO_CONDA_DIR/envs/ObjectDetectionTutorial_TF2/lib/python3.7/site-packages/tensorflow/models/research/object_detection/object_detection_tutorial/fresh_new_images" --threshold 0.8
```

## Want to start over?
If something went wrong and you want to start over, just execute `conda env remove --name ObjectDetectionTutorial_TF2` in a new terminal window. Then you can start again at step 1. Be careful though, this will delete the entire `ObjectDetectionTutorial_TF2` folder. So make sure you have copies of all the required files, such as your training images and labels. 
