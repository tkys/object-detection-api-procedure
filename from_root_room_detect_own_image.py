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

#PATH_TO_FROZEN_GRAPH="/home/user1/workspace/models/research/object_detection/inference_graph/ssd_resnet50_fpn_coco/frozen_inference_graph.pb"
#PATH_TO_FROZEN_GRAPH="/home/user1/workspace/models/research/object_detection/inference_graph/faster_rcnn_inception_resnet_v2_atrous_coco/frozen_inference_graph.pb"
#PATH_TO_FROZEN_GRAPH="/home/user1/workspace/models/research/object_detection/inference_graph/faster_rcnn_nas_coco/frozen_inference_graph.pb"
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
