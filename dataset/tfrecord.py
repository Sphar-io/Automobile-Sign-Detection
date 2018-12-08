import tensorflow as tf
from object_detection.utils import dataset_util
import base64
import random
import pandas
from PIL import Image
import requests
from io import BytesIO
import shutil
import os

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('json_input', '', 'Path to the JSON input')
flags.DEFINE_string('train_output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('test_output_path', '', 'Path to output TFRecord')
# flags.DEFINE_string('create_eval', 'false', 'whether to output all images for testing')
FLAGS = flags.FLAGS
validCategories = ["stop","stopSign","speedLimit","speedLimit15", "speedLimit25", "speedLimit30", "speedLimit35", "speedLimit40", "speedLimit45", "speedLimit50", "speedLimit55", "speedLimit65"]
goalDim = 300


def id_from_category(category):
  if(category in validCategories):
    if category == "stop" or category == "stopSign":
      return ("stopSign",2)
    else:
      return ("speedLimit",1) 
  else:
    return (-1,-1)

def create_tf_lisa_example(rows,toSave):
  classes = rows.iloc[:, 1].tolist()
  if not any(cat in validCategories for cat in classes):
    return -1
 
  imgpath = './' + rows.iloc[0][0]

  im = Image.open(imgpath)

  w, h = im.size
  im = im.crop(( w-h, 0, w, h ))
  im = im.resize((goalDim,goalDim), Image.ANTIALIAS)

  xTranslate = w-h

  if(toSave):
    filepath = rows.iloc[0][0].split('/')
    im.convert('RGB').save("eval_set\\" + filepath[0] + filepath[2] + ".jpg","JPEG")

  im.convert('RGB').save(".\\tmp.jpg","JPEG")

  with tf.gfile.GFile(".\\tmp.jpg", 'rb') as fid:
    encoded_image_data = fid.read()

  image_format = b'jpg' # b'jpeg' or b'png'
  
  width, height = im.size
  filename = str.encode(rows.iloc[0][0])
  
  xmins = []
  xmaxs = []
  ymins = []
  ymaxs = []
  classes_text = []
  classes = []

  for index, row in rows.iterrows():
    category, idx = id_from_category(row[1])
    if category == -1: # this sign is not a speed or stop sign
      continue
    if row[6] == 1: # this sign is occluded
      continue

    xloc = (int(row[2]) - xTranslate)
    # print(xloc)
    if(xloc < 0): 
        continue

    xmins.append(min(1, int(row[2] - xTranslate) / h)) # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs.append(min(1, int(row[4] - xTranslate) / h)) # List of normalized right x coordinates in bounding box

    ymins.append(min(1, int(row[3]) / h)) # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs.append(min(1, int(row[5]) / h)) # List of normalized bottom y coordinates in bounding box

    classes_text.append(str.encode(category)) # List of string class name of bounding box (1 per box)
    classes.append(idx) # List of integer class id of bounding box (1 per box)

  if(len(xmins) == 0):
    return -1

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example

def handle_lisa_set(testWriter, trainWriter):
  csv_df = pandas.read_csv(FLAGS.csv_input,sep=';')
  csv_df = csv_df.sample(frac=1).reset_index(drop=True)
  print(len(csv_df.index))
  csv_df = csv_df[csv_df['Filename'].str.contains("vid")]
  print(len(csv_df.index))
  groups_df = csv_df.groupby(["Origin file"])

  totalNum = len(csv_df.index)
  curNum = 1
  trainEx = 0
  testEx = 0

  for groupId , values in groups_df:
    curNum = curNum + len(values.index)
    if(curNum % 50 == 0):
        print(100*(curNum / totalNum)," percent done")

    trainOrTest = random.random()
    # how to save test (validation) images 
    if trainOrTest > .1:
      tf_example = create_tf_lisa_example(values,False)
      if tf_example == -1:
        continue
      trainEx = trainEx + 1
      trainWriter.write(tf_example.SerializeToString())
    else:
      tf_example = create_tf_lisa_example(values,True)
      if tf_example == -1:
        continue
      testEx = testEx + 1
      testWriter.write(tf_example.SerializeToString())

  print("total LISA images processed", curNum)
  print("Training set: ", trainEx)
  print("Testing set", testEx)


def create_tf_dataturks_example(rows, toSave):
  imgpath = rows["content"]
  
  response = requests.get(imgpath)
  im = Image.open(BytesIO(response.content))
  
  filename = imgpath.split('/')[4]

  w, h = im.size
  im = im.crop(( w-h, 0, w, h ))

  xShift = (w-h)
  im = im.resize((goalDim,goalDim), Image.ANTIALIAS)


  if(toSave):
    im.save("eval_set\\" + filename,"JPEG")

  im.save("tmp.jpg","JPEG")

  with tf.gfile.GFile(".\\tmp.jpg", 'rb') as fid:
    encoded_image_data = fid.read()

  image_format = b'jpg' # b'jpeg' or b'png'
  
  width, height = im.size
  filename = str.encode(filename)
  
  xmins = []
  xmaxs = []
  ymins = []
  ymaxs = []
  classes_text = []
  classes = []

  for row in rows["annotation"]:
    # print(row["points"])
    category, idx = id_from_category(row["label"][0])
    if category == -1: # this sign is not a speed or stop sign
      print("ERROR")
      continue

    points = row["points"]
    xloc = points[0][0]*w - xShift
    if(xloc < 0): 
        continue

    xmins.append(min(1, (points[0][0]*w - xShift)/h)) # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs.append(min(1, (points[2][0]*w - xShift)/h)) # List of normalized right x coordinates in bounding box

    ymins.append(min(1, points[0][1])) # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs.append(min(1, points[1][1])) # List of normalized bottom y coordinates in bounding box

    classes_text.append(str.encode(category)) # List of string class name of bounding box (1 per box)
    classes.append(idx) # List of integer class id of bounding box (1 per box)
  
  if(len(xmins) == 0):
    return -1

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example

def handle_dataturks_set(testWriter, trainWriter):
  json_df = pandas.read_json(FLAGS.json_input, orient='columns',lines=True)
  json_df.dropna(subset=['annotation'], inplace = True)

  totalNum = len(json_df.index)
  curNum = 1
  trainEx = 0
  testEx = 0
  
  # Shuffles the dataset
  json_df = json_df.sample(frac=1).reset_index(drop=True)

  for index, row in json_df.iterrows():
    curNum = curNum + 1
    if(curNum % 50 == 0):
        print(100*(curNum / totalNum)," percent done")

    trainOrTest = random.random()
    if trainOrTest > .1:
      tf_example = create_tf_dataturks_example(row, False)
      if tf_example == -1:
        continue
      trainEx = trainEx + 1
      trainWriter.write(tf_example.SerializeToString())
    else:
      tf_example = create_tf_dataturks_example(row,True)
      if tf_example == -1:
        continue
      testEx = testEx + 1
      testWriter.write(tf_example.SerializeToString())

  print("total DataTurk images processed", curNum)
  print("Training set: ", trainEx)
  print("Testing set", testEx)


def main(_):
  shutil.rmtree(".\\eval_set")
  os.makedirs(".\\eval_set")

  trainWriter = tf.python_io.TFRecordWriter(FLAGS.train_output_path)
  testWriter = tf.python_io.TFRecordWriter(FLAGS.test_output_path)
  
  handle_lisa_set(testWriter, trainWriter)
  handle_dataturks_set(testWriter, trainWriter)

  trainWriter.close()
  testWriter.close()

if __name__ == '__main__':
  tf.app.run()