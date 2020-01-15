# object-detection-api-procedure


フォトラクション
detection api でのfinetune　独自データでの学習、推論モデルのフロー


今回はfaster rcnn で

##################################
Object Detection APIのリポジトリをクローンします。

$ git clone https://github.com/tensorflow/models
$ cd model/research

##########################
## pathを通しておく
# From tensorflow/models/research/

$ cd model/research
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
$ echo $PYTHONPATH
:/home/spi/tk_yoshikawa/photolac/models/research:/home/spi/tk_yoshikawa/photolac/models/research/slim

##########################
Protobuf Compilation
The Tensorflow Object Detection API uses Protobufs to configure model and training parameters. Before the framework can be used, the Protobuf libraries must be compiled. This should be done by running the following command from the tensorflow/models/research/ directory:

Download and install the 3.0 release of protoc, then unzip the file.

# From tensorflow/models/research/

$ wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
$ unzip protobuf.zip
Run the compilation process again, but use the downloaded version of protoc

$ ./bin/protoc object_detection/protos/*.proto --python_out=.

#ファイルが生成される


##################################

## 学習用の画像収集

サイズは特に、指定無し 
./images ディレクトリ以下へ配置しておく

（参考）
'./images/100.png'
'./images/200.png'

##################################
## 学習様に画像へ矩形アノテーションを行い、XY座標の入ったデータを準備

.csv形式
カラム名は 'class','img_path', 'xmin', 'ymin', 'xmax', 'ymax'

（参考）
class,filename,xmin,ymin,xmax,ymax
room,100.png,2376,649,2405,667
room,100.png,2066,584,2125,609
room,100.png,1742,523,1837,598
room,200.png,1550,533,1621,560
room,200.png,1236,616,1293,643


##学習用、テスト評価用に2種作成しておく
./labels ディレクトリ以下へ配置しておく
（参考）
'./labels/train_labels.csv'
'./labels/test_labels.csv'


##################################
##Make a new file object-detection.pbtxt which looks like this:
item {
  id: 1
  name: 'room'
}


##################################
## label.csvからtfrecord形式へ変換する
generate_tfrecord.py 

# From tensorflow/models/reserch/
$ cd ./tensorflow/models/reserch/

# Create train data:
$ python generate_tfrecord.py --csv_input=./path/to/train_labels.csv  --output_path=/path/to/train.record --image_dir ../images


# Create test data:
$  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record --image_dir ../images

##################################
## fine-tuneする学習済みモデルを決め、configを書き直す

今回はfaster rcnn でいく

$ cd ./models/research/object_detection/faster_rcnn_nas_coco/
$ cp ./pipeline.config ./pipeline.config_backup

編集する

#before
num_classes: 90
#after
num_classes: 1

#before
height: 1200
width: 1200
#after
height: 300
width: 300

#before
schedule {
  step: 0
  learning_rate: 0.000300000014249
}

#after
#ステップごとの学習率スケジュール　initial_learning_rate:が既に指定されているので、step:0箇所でも指定されると学習時にエラー出る。コメントアウトした
#schedule {
#  step: 0
#  learning_rate: 0.000300000014249
#}


#before
  gradient_clipping_by_norm: 10.0
  fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"
  from_detection_checkpoint: true
  num_steps: 200000
}
train_input_reader {
  label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"
  tf_record_input_reader {
    input_path: "PATH_TO_BE_CONFIGURED/mscoco_train.record"
  }
}
eval_config {
  num_examples: 8000
  max_evals: 10
  use_moving_averages: false
}
eval_input_reader {
  label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"
  shuffle: false
  num_readers: 1
  tf_record_input_reader {
    input_path: "PATH_TO_BE_CONFIGURED/mscoco_val.record"
  }
}

#after
  gradient_clipping_by_norm: 10.0
  fine_tune_checkpoint: "faster_rcnn_nas_coco/model.ckpt"
  from_detection_checkpoint: true
  num_steps: 200000
}
train_input_reader {
  label_map_path: "data/object-detection.pbtxt"
  tf_record_input_reader {
    input_path: "data/train.record"
  }
}
eval_config {
  num_examples: 8000
  max_evals: 10
  use_moving_averages: false
}
eval_input_reader {
  label_map_path: "data/object-detection.pbtxt"
  shuffle: false
  num_readers: 1
  tf_record_input_reader {
    input_path: "data/test.record"
  }
}

###学習進捗を確認

tensorboard --logdir=/tmp/tmp*******


#### 学習したcheckpointから推論用モデル（.pb形式）データを生成する

cd ./models/research/object_detection/

python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./faster_rcnn_nas_coco/pipeline.config  --trained_checkpoint_prefix ./tmplog/fast_rcnn_nas_coco/tmp6q8zwuzm/model.ckpt-3469  --output_directory  ./inference_graph/faster_rcnn_nas_coco/

# --input_type image_tensor  #固定
# --pipeline_config_path #学習時に指定したconfigファイル}
# --trained_checkpoint_prefix  # 学習時に出力されたckptデータ群指定　# model.ckpt-3469 #ここまで書けば良い
# --output_directory # 推論用データの出力先ディレクトリ　


python export_inference_graph.py --input_type image_tensor --pipeline_config_path ../../../ssd_resnet50_fpn_coco_1221_batch4_step100000/pipeline.config --trained_checkpoint_prefix ../../../tmpf6bj2q6n/model.ckpt-63267 --output_directory  ./output_inference/


# 以下ファイルが生成される
frozen_inference_graph.pb
model.ckpt.meta
model.ckpt.data-00000-of-00001
model.ckpt.index
checkpoint
pipeline.config
saved_model

./inference_graph など適当なディレクトリを作ってそこに上記を全部入れておく

注意）.pbファイルを生成した際と後の検出実行する際のtensorflowバージョンは合わせる必要ないとエラーになる。（今回は==1.13.1）


ここからこのモデルを使って他の画像を検出する
from_root_room_detect_own_image.py

=====
#!/usr/bin/env python
# coding: utf-8


import sys

from pathlib import Path
base_path =Path(__file__).resolve().parent

MODEL_BASE = str(base_path) + '/models/research'
#print(MODEL_BASE)

sys.path.append(MODEL_BASE)
sys.path.append(MODEL_BASE + '/object_detection')
sys.path.append(MODEL_BASE + '/slim')
sys.path.append(MODEL_BASE + '/object_detection/utils')


# In[1]:

import tensorflow as tf
#tf.__version__


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# In[2]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import matplotlib
matplotlib.use('Agg')

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")
#print(sys.path)

from object_detection.utils import ops as utils_ops

from utils import label_map_util

from utils import visualization_utils as vis_util

args = sys.argv

#print(args[0])
#print(args[1])
#print(args[2])

# In[3]:


#PATH_TO_FROZEN_GRAPH="out/frozen_inference_graph.pb"

#PATH_TO_FROZEN_GRAPH="/home/spi/tk_yoshikawa/photolac/models/research/object_detection/inference_graph/ssd_resnet50_fpn_coco/frozen_inference_graph.pb"
#PATH_TO_FROZEN_GRAPH="/home/spi/tk_yoshikawa/photolac/models/research/object_detection/inference_graph/faster_rcnn_inception_resnet_v2_atrous_coco/frozen_inference_graph.pb"
#PATH_TO_FROZEN_GRAPH="/home/spi/tk_yoshikawa/photolac/models/research/object_detection/inference_graph/faster_rcnn_nas_coco/frozen_inference_graph.pb"
#PATH_TO_FROZEN_GRAPH="./output_inference/frozen_inference_graph.pb"
#PATH_TO_FROZEN_GRAPH="./output_inference/ssd_resnet50_fpn_coco_1221_batch4_step100000/frozen_inference_graph.pb"
#PATH_TO_FROZEN_GRAPH= MODEL_BASE + '/object_detection/' + "/output_inference/ssd_resnet50_fpn_coco_1221_batch4_step100000/frozen_inference_graph.pb"
PATH_TO_FROZEN_GRAPH= "./pbfile/frozen_inference_graph.pb"


#PATH_TO_LABELS="training/labelmap.pbtxt"
#PATH_TO_LABELS = './object-detection.pbtxt'
#PATH_TO_LABELS =  MODEL_BASE + '/object_detection/' + '/object-detection.pbtxt'
PATH_TO_LABELS =  "./object-detection.pbtxt"



# In[4]:

print("----Londing Model...")
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef() #tf.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
print("----Load Model DONE!")

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)



def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)



# In[5]:


#get_ipython().system(' python -V')


# In[6]:


import os
import pathlib

#PATH_TO_TEST_IMAGES_DIR = 'images'
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'train/IMG_20190222_224827.jpg')]

#PATH_TO_TEST_IMAGES_DIR = pathlib.Path('./image_eval/')
#TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.png")))
TEST_IMAGE_PATHS = [args[1]]
print(TEST_IMAGE_PATHS)
print(os.path.basename(args[1]))


# Size, in inches, of the output images.
IMAGE_SIZE = (60, 40)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.compat.v1.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.compat.v1.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference

      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


for image_path in TEST_IMAGE_PATHS:
  image = Image.open(image_path)
  #print(image.filename)


  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)


  # Actual detection.
  print("----Now Predicting..")
  output_dict = run_inference_for_single_image(image_np, detection_graph)



  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=4,
      min_score_thresh=.2
      )
  plt.figure(figsize=IMAGE_SIZE)
  plt.imshow(image_np)
  plt.savefig('result_' + os.path.basename(args[1]) )
  print("----Save Result Image  : %s" % str('result_' + os.path.basename(args[1])) )


# In[7]:


#TEST_IMAGE_PATHS


# In[8]:

output = output_dict['detection_boxes'][0:int(output_dict['num_detections'])]


import json
print(json.dumps(output.tolist()))

======
