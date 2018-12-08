# Automobile Sign Detection
This repository contains the necessary pieces to train your own Neural Network to recognize speed limit and stop signs.

Uses Tenorflow's Mobilenet v2 SSD, and the UCSD LISA dataset (Supplemented with a custom annotated dataset). 

## Usage

### Dataset
- Install Tensorflow and Tensorboard
- Download a copy of the LISA Dataset (link below) and copy [/vid0-vid10, allAnnotations.csv] to /data 
- Assemble TF Record files for ingestion by Tensorflow using dataset/tfRecord.py. This will also generate an /eval_set folder for running manual model verification later. (see commands.txt for example)
- Inspect the quality of your tfRecord using dataset/inspector.py (Only displays one annotation per frame)
- Copy the testing and training records to /data

### Training
- Clone the Tensorflow models directory (link below) to the root repository directory
- Download a copy of the Mobilenet architecture from the Tensorflow link below and place in the models\model folder (model.cpkt.[data, index, meta])
- Bash run.bat will begin training
- Run tensorboard --logdir=.\models\model to see training updates 

### Finishing up
- Run the Export Graph command from commands.txt to save your trained model.

## Notes
It is highly recomended to use a Python Environment and package manager to simplify installing and conflict resolution. I used Miniconda.

Some paths are hard-coded due to varying levels of interoperability with Window's path string backslash. Expect to have to update these, especially if runing on Unix.

## Downloads
- https://github.com/tensorflow/models
- http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html

## Resources
- https://github.com/tensorflow/models/tree/master/research/object_detection
- https://towardsdatascience.com/detecting-pikachu-on-android-using-tensorflow-object-detection-15464c7a60cd
