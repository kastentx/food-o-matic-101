# app.py
#
# A simple example of hosting a TensorFlow model as a Flask service
#
# Copyright 2017 ActiveState Software Inc.
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

import logging
import random
import time

from flask import Flask, jsonify, request

import numpy as np
import tensorflow as tf

import requests
from io import BytesIO
from PIL import Image

app = Flask(__name__)

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=224, input_width=224,
				input_mean=128, input_std=128):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def read_tensor_from_image_url(loaded_image, image_url, input_height=224, input_width=224,
				input_mean=128, input_std=128):
  input_name = "file_reader"
  output_name = "normalized"
  #file_reader = tf.read_file(file_name, input_name)
  if image_url.endswith(".png"):
    image_reader = tf.image.decode_png(loaded_image, channels = 3,
                                       name='png_reader')
  elif image_url.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(loaded_image,
                                                  name='gif_reader'))
  elif image_url.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(loaded_image, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(loaded_image, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
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

@app.route('/')
def test_route():
  return jsonify("Hello Flask.")

@app.route('/predict_web')
def classify_web():
    # load image from url
    image_url = request.args['url']
    #image_data = Image.open(requests.get(image_url, stream=True).raw)

    # try it with passing the string data instead of actual image
    image_data = requests.get(image_url, stream=True).content

    image_tensor = read_tensor_from_image_url(image_data, image_url)

    # process the image so its valid input
    #float_caster = tf.cast(image_tensor, tf.float32)
    #dims_expander = tf.expand_dims(float_caster, 0);
    #resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    #normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    #sess = tf.Session()
    #result = sess.run(normalized)

    # make prediction
    with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: image_tensor})
        end=time.time()
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(label_file)

    print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))

    for i in top_k:
        print(labels[i], results[i])

    return jsonify(labels,results.tolist())

@app.route('/predict')
def classify():
    file_name = request.args['file']

    t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)
        
    with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
        end=time.time()
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(label_file)

    print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))

    for i in top_k:
        print(labels[i], results[i])

    return jsonify(labels,results.tolist())

# GOTTA HAVE THOSE GLOBALS DEFINED YALL

# TensorFlow configuration/initialization
model_file = "retrained_graph.pb"
label_file = "retrained_labels.txt"
input_height = 224
input_width = 224
input_mean = 128
input_std = 128
input_layer = "Placeholder"
output_layer = "final_result"

# Load TensorFlow Graph from disk
graph = load_graph(model_file)

# Grab the Input/Output operations
input_name = "import/" + input_layer
output_name = "import/" + output_layer
input_operation = graph.get_operation_by_name(input_name);
output_operation = graph.get_operation_by_name(output_name);