# Automobile Sign Detection
This repository contains the necessary pieces to train your own Neural Network to recognize speed limit and stop signs.

Uses Tenorflow's Mobilenet v2 SSD, and the UCSD LISA dataset (Supplemented with a custom annotated dataset).

## Overview
- Install Tensorflow and Tensorboard
- Assemble TF Record files for ingestion by Tensorflow using dataset/tfRecord.py.
- Inspect the quality of your tfRecord using dataset/inspector.py (Only displays one annotation per frame)
- Copy your testing and training records to /data
- Download a copy of the Mobilenet architecture from the Tensorflow link below and place in the models\model folder
- Bash run.bat will begin training! run tensorboard --logdir=.\models\model to see training updates 


## Resources
- https://github.com/tensorflow/models/tree/master/research/object_detection
- https://towardsdatascience.com/detecting-pikachu-on-android-using-tensorflow-object-detection-15464c7a60cd
