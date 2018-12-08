import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

record_iterator = tf.python_io.tf_record_iterator(path='.\\data\\testData.tfrecords')

model = tf.global_variables_initializer()
sess = tf.Session()
sess.run(model)
cur = 0;
for string_record in record_iterator:
    cur = cur + 1
    if(cur % 2 == 0): continue

    example = tf.train.Example()
    example.ParseFromString(string_record)
    
    height = int(example.features.feature['image/height']
                                 .int64_list
                                 .value[0])
    
    width = int(example.features.feature['image/width']
                                .int64_list
                                .value[0])

    filename = (example.features.feature['image/filename']
                                  .bytes_list
                                  .value[0])
    
    img_string = (example.features.feature['image/encoded']
                                  .bytes_list
                                  .value[0])
    

    xmin = float(example.features.feature['image/object/bbox/xmin']
                                 .float_list.value[0])

    ymin = float(example.features.feature['image/object/bbox/ymin']
                                 .float_list
                                 .value[0])
    xmax = float(example.features.feature['image/object/bbox/xmax']
                                 .float_list
                                 .value[0])
    ymax = float(example.features.feature['image/object/bbox/ymax']
                                 .float_list
                                 .value[0])

    
    classType = (example.features.feature['image/object/class/text']
                                  .bytes_list)  

    classID = (example.features.feature['image/object/class/label']
                                  .int64_list)
    
    # classID = json.loads(classID)

    rectWidth = (xmax - xmin)*width
    rectHeight = (ymax - ymin)*height
    xmin = xmin*width
    xmax = xmax*width
    ymin = ymin*height
    ymax = ymax*height

    rect = patches.Rectangle((xmin,ymin),rectWidth,rectHeight,linewidth=1,edgecolor='r',facecolor='none')

    img_decode = tf.image.decode_image(img_string)

    fig,ax = plt.subplots(1)
    
    result = sess.run(img_decode)
    ax.imshow(result)
    ax.add_patch(rect)
    plt.show()