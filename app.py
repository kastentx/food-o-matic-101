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

app = Flask(__name__)

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_url(image_data, image_url, input_height=224, input_width=224,
				input_mean=128, input_std=128):
  input_name = "file_reader"
  output_name = "normalized"
  if image_url.endswith(".png"):
    image_reader = tf.image.decode_png(image_data, channels = 3,
                                       name='png_reader')
  elif image_url.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(image_data,
                                                  name='gif_reader'))
  elif image_url.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(image_data, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(image_data, channels = 3,
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

def get_ranking(top_5_labels_with_scores):
  x = top_5_labels_with_scores
  sorted_x = sorted(x, key=x.get, reverse=True)
  return {rank: key for rank, key in enumerate(sorted_x, 1)}

@app.route('/')
def test_route():
  return jsonify("Hello Flask.")

@app.route('/predict_web')
def classify_web():
    # load image from url
    image_url = request.args['url']
    image_data = requests.get(image_url, stream=True).content
    image_tensor = read_tensor_from_image_url(image_data, image_url)

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

    top_5_rank = {}
    top_5_score = {}
    for i in top_k:
        print(labels[i], results[i])
        top_5_score[labels[i]] = np.float64(results[i]).item()
        top_5_rank = get_ranking(top_5_score)


    return jsonify(rank=top_5_rank, score=top_5_score)

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

if __name__ == '__main__':
  app.run(debug=True)