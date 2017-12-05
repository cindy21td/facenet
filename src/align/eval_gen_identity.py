"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import random
import math
import pickle
from time import sleep


def main(args):
    dataset = facenet.get_dataset(args.input_dir)
    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    nrof_images_total = 0
    nrof_successfully_aligned = 0

    filtered_dct = {}

    if args.random_order:
        random.shuffle(dataset)
    for cls in dataset:
        filtered_dct[cls.name] = []
        for image_path in cls.image_paths:
            nrof_images_total += 1
            try:
                img = misc.imread(image_path)
            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(image_path, e)
                print(errorMessage)
            else:
                if img.ndim < 2:
                    print('Unable to align "%s"' % image_path)
                    ## Count this as alignment error
                    continue
                if img.ndim == 2:
                    img = facenet.to_rgb(img)
                img = img[:, :, 0:3]

                bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                ## Accept only 1 face!
                if nrof_faces == 1:
                    ## Alignment success, .. now to classify the images.
                    nrof_successfully_aligned += 1
                    filtered_dct[cls.name].append(image_path)
                else:
                    print('Unable to align "%s"' % image_path)
                    ## Count this as alignment error
                    continue


    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


    ## By this point our dataset is filtered to only contain images with detected faces.
    filtered_dataset = [facenet.ImageClass(class_name, image_paths) for class_name, image_paths in filtered_dct.items()]

    ## Alignment accuracy
    accuracy = 1. * nrof_successfully_aligned / nrof_images_total
    print('Aligned accuracy: %.3f' % accuracy)
    ## Accuracy for full dataset
    print('Identity accuracy for full test set: %.3f' % test_identity(args, dataset))
    ## Accuract for filtered dataset
    print('Identity accuracy for successful aligned images: %.3f' % test_identity(args, filtered_dataset))

def test_identity(args, dataset):
    with tf.Graph().as_default():

        with tf.Session() as sess:
            np.random.seed(seed=args.seed)
            # Classify images
            print('Testing classifier')
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            paths, labels = facenet.get_image_paths_and_labels(dataset, class_names)

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            print('Num epoch: ', nrof_batches_per_epoch)

            for i in range(nrof_batches_per_epoch):
                print('Epoch: ', i)
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

            print('Loaded classifier model from file "%s"' % classifier_filename_exp)

            predictions = model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

            for i in range(len(best_class_indices)):
                print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

            accuracy = np.mean(np.equal(best_class_indices, labels))
            print(best_class_indices, labels)
            return accuracy

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename',
                        help='Classifier model file name as a pickle (.pkl) file. ' +
                             'For training this is the output and for classification this is an input.')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--random_order',
                        help='Shuffles the order of images to enable alignment using multiple processes.',
                        action='store_true')
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
