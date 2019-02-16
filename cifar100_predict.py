import tensorflow as tf
import os
from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np
import scipy
from scipy import misc
import matplotlib.pyplot as plt
import cv2


def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    return x


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph


def main():

    graph = load_graph("serve/cifar100_inference/1/saved_model.pb")
    print("Load Image...")
    bgr_img = cv2.imread('apple.jpg')
    b, g, r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r, g, b])
    resize = cv2.resize(rgb_img, (32, 32), interpolation=cv2.INTER_AREA)
    image = normalize(resize)

    for op in graph.get_operations():
        print(op.name)

    x = graph.get_tensor_by_name('prefix/Placeholder/input_x:0')
    y = graph.get_tensor_by_name('prefix/logits/BiasAdd:0')
    y_with_softmax = tf.nn.softmax(y)

    with tf.Session(graph=graph) as sess:
        y_out = sess.run(y_with_softmax, feed_dict={x: image})
        print(y_out)


if __name__ == '__main__':
    main()
