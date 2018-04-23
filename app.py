# app.py
#
# Serves a MobileNet based 
# food image-recognition model as a Flask service
#
# Inspired by / Adapted from 'tensorflask'
# by The TensorFlow Authors and ActiveState.
#
# https://github.com/ActiveState/tensorflask
#
# Trained using the food-101 dataset 
# https://www.vision.ee.ethz.ch/datasets_extra/food-101/
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

import time
import numpy as np
import tensorflow as tf
import requests

from flask import Flask, jsonify, request

app = Flask(__name__)

def load_graph(path_to_model):
    tf_graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(path_to_model, "rb") as model_file:
        graph_def.ParseFromString(model_file.read())
    with tf_graph.as_default():
        tf.import_graph_def(graph_def)
    return tf_graph

def read_tensor_from_image_url(image_data, image_url, height=224, width=224, mean=128, std=128):
    if image_url.endswith(".png"):
        image_reader = tf.image.decode_png(image_data, channels=3, name='png_reader')
    elif image_url.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(image_data, name='gif_reader'))
    elif image_url.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(image_data, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(image_data, channels=3, name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [height, width])
    normalized = tf.divide(tf.subtract(resized, [mean]), [std])
    sess = tf.Session()
    result = sess.run(normalized)
    return result

def load_labels(path_to_labels):
    labels = []
    proto_as_ascii_lines = tf.gfile.GFile(path_to_labels).readlines()
    for label in proto_as_ascii_lines:
        labels.append(label.rstrip())
    return labels

def get_ranking(top_5_labels_scores):
    sorted_labels = sorted(top_5_labels_scores, key=top_5_labels_scores.get, reverse=True)
    return {rank: key for rank, key in enumerate(sorted_labels, 1)}

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
    loaded_graph = load_graph("retrained_graph.pb")
    with tf.Session(graph=loaded_graph) as sess:
        input_layer = "Placeholder"
        output_layer = "final_result"
        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = loaded_graph.get_operation_by_name(input_name)
        output_operation = loaded_graph.get_operation_by_name(output_name)
        start = time.time()
        results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: image_tensor})
        end = time.time()
        results = np.squeeze(results)
        top_k = results.argsort()[-5:][::-1]
        labels = load_labels("retrained_labels.txt")
    print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
    top_5_score = {}
    for i in top_k:
        print(labels[i], results[i])
        top_5_score[labels[i]] = np.float64(results[i]).item()
    return jsonify(rank=get_ranking(top_5_score), score=top_5_score, \
                   eval_time='{:.3f}s'.format(end-start))

if __name__ == '__main__':
    app.run(debug=True)
