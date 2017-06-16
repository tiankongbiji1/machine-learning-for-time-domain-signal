# machine-learning-for-time-domain-signal
machine learning
#trains and and evaluates the mnist network using a feed dictionary
from_future_import absolute_import
from_future_import division
from_future_import print_function

import os.path
import time

import tensorflow.python.platform
import numpy
from six.moves import xrange
import tensorflow as tf

# Basic model parameters as external flags
flag=tf.app.flags
FLAGS=flags.FLAGS
flags.DEFINE_float('learning_rate',0.01,'Initial learing rate.')
flags.DEFINE_integer('max_steps',2000,'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1',128,'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2',32,'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size',100,'Batch size. ','Must divide evenly into the dataset size.')
flags.DEFINE_string('train_dir','Mnist_data/','Directory to put the training data.')
flags.DEFINE_boolean('fake_data',False,'If true, uses fake data ','for unit testing.')

def placeholder_input(batch_size):
 images_placeholder=tf.placeholder(tf.float32,shape=(batch_size,image_pixel))
 labels_placeholde=tf.placeholder(tf.int32,shape=(batch_size))
 return images_placeholder,label_placeholder

def fill_feed_dict(data_set,images_pl,labels_Pl):
 images_feed,labels_feed=data_set.next_batch(FLAGS.batch_size,FLAGS.fake_size)
 feed_dict={
  image_pl:images_feed,
  labels_pl:labels_feed,
 }
 return feed_dict
 
 def do_eval(sess,
             eval_correct,
             images_placeholder,
             labels_placeholder,
             data_set):
 true_count=0
 steps_per_epoch=data_set.num_examples
 num_example=steps_per_epoch*FLAGS.batch_size
 for step in xrange(steps_per_epoch):
  feed_dict=fill_feed_dict(data_set,
                           images_placeholder,
                           labels_placeholder)
  true_count+=sess.run(eval_correct,feed_dict=feed_dict)
 precision=true_count/num_examples
 print(' Num examples:%d N
