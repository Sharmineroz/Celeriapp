# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import os

import numpy as np
import tensorflow as tf
import cv2
#import pyrealsense2 as rs

DIM = (640,480)
K = np.array([[475.53765869140625, 0.0, 309.2130126953125], 
            [0.0, 475.5375061035156, 245.9634552001953], 
            [0.0, 0.0, 1.0]])

D = np.array([[0.14829276502132416], 
            [0.06511270254850388], 
            [0.004560006316751242],
            [0.0025975550524890423]])
camx, camy, fps=640, 480, 30

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_img(img, input_height=299, input_width=299,
        input_mean=0, input_std=255):
  
  float_caster = tf.cast(img,tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


if __name__ == "__main__":

  path=os.getcwd()
  
  # MobileNet
  net_width=50
  input_height = 224
  input_width = 224
  input_mean = 128
  input_std = 128
  input_layer = "input"

  ## Inception
  # input_height = 299
  # input_width = 299
  # input_mean = 0
  # input_std = 255
  # input_layer = "Mul"
  output_layer = "final_result"

  model_file = "../mods/retrained_graph_"+str(input_height)+"_0"+str(net_width)+".pb"
  label_file = "../mods/retrained_labels.txt"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = load_graph(model_file)

  ##Jetson Onboard camera
  cap=cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);

  labels = load_labels(label_file)

  time_array = np.zeros(30, dtype=np.float)

  while True:

    #frames = pipeline.wait_for_frames()
    #depth_frame = frames.get_depth_frame()
    #color_frame = frames.get_color_frame()

    #if not depth_frame or not color_frame:
    #  continue

    # Convert images to numpy arrays
    #depth_image = np.asanyarray(depth_frame.get_data())
    #frame = np.asanyarray(color_frame.get_data())
    
    ret, frame = cap.read()
    if not ret:
      continue	
    
    #frame = add_background(frame, depth_image)
    
    t = read_img(frame,
                 input_height=input_height,
                 input_width=input_width,
                 input_mean=input_mean,
                 input_std=input_std)
    

    #config = tf.ConfigProto()
    #config.intra_op_parallelism_threads = 4
    #config.inter_op_parallelism_threads = 4

    with tf.Session(graph=graph) as sess:
      start = time.time()
      results = sess.run(output_operation.outputs[0],
                        {input_operation.outputs[0]: t})
      end=time.time()
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]

    time_array[1:time_array.shape[0]]=time_array[0:time_array.shape[0]-1]
    time_array[0]= (end-start)

    print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
    template = "{} (score={:0.5f})"
    for i in top_k:
      print(template.format(labels[i], results[i]))

    cv2.imshow('frame',frame.astype(np.uint8))

    if cv2.waitKey(1) & 0xFF == ord('1'):
    	np.save("../ev_time/time_array"+str(input_height)+"_0"+str(net_width),time_array)
    	break
  cap.release()
  cv2.destroyAllWindows()
    
