#Importing all needed packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
import argparse

import numpy as np
#import pandas as pd
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Conv3D,MaxPool2D,Reshape,Input,Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.models import Model
import horovod.tensorflow as hvd
#import horovod.tensorflow.keras as hvd
from azureml.core.run import Run

class Parameters:
    def __init__(self):
        self.pixels = 100
        self.classes = 3
        self.batch_size = 64
        self.epochs = 10
        self.num_batches=20

params = Parameters()
# Class definitions
cls = {'Normal':1,'bacteria':2,'virus':3}


def model_fn(features,labels,mode):
    
    print("IN model_fn",features.get_shape())
    
    #Combine it with keras
    model_input = Input(tensor=features)

    #Build your network
    conv1 = Conv2D(512,(5,5),activation=tf.nn.relu)(model_input)
    max_pool = MaxPool2D(pool_size=(2,2))(conv1)
    conv2 = Conv2D(128,(2,2),activation=tf.nn.relu)(max_pool)
    max_pool2 = MaxPool2D(pool_size=(2,2))(conv2)
    flatten = Flatten(input_shape=(-1, params.pixels, params.pixels, 1))(max_pool2)
    full_connected = Dense(1000)(flatten)
    drop = Dropout(0.25)(full_connected)
    y_pred = Dense(params.classes)(drop)
    predictions = {"classes":tf.argmax(input=y_pred, axis=1),
                  "probabilities":tf.nn.softmax(y_pred, name="softmax_tensor")}

    # Check if mode is PREDICT
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)
     
  
    # Define loss & optimizer, minimize cross entropy
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32),
                               depth=params.classes)
    
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                           logits=y_pred)
    
    #Log metrics
    run = Run.get_context()
    
   
    if mode == tf.estimator.ModeKeys.TRAIN:
        print('Training mode....')
        lr = 0.001
        
        # Horovod: scale learning rate by the number of workers.
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr * hvd.size(), momentum=0.9)

        # Horovod: add Horovod Distributed Optimizer.
        optimizer = hvd.DistributedOptimizer(optimizer)

        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    
    # Add evaluation metrics (for EVAL mode)
    
    eval_metric_ops = {"Accuracy": tf.metrics.accuracy(labels=tf.argmax(labels,axis=1), predictions=predictions["classes"])}
    #print("Eval loss:",eval_metric_ops["Accuracy"])
    #run.log('loss',eval_metric_ops["Accuracy"])
    
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    
def _batch_normalization(tensor, epsilon=.0001):
    mean,variance = tf.nn.moments(tensor,axes=[0])
    return((tensor-mean)/(variance+epsilon))

def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    features = {'image': tf.FixedLenFeature([], tf.string),
                        "label": tf.FixedLenFeature([], tf.int64)}
    
    # Load one example
    features = tf.parse_single_example(proto, features)
    
    # Turn your saved image string into an array
    features['image'] = tf.decode_raw(features['image'], tf.uint8)
    
    features['image'] = tf.cast(features['image'],tf.float32)
    
    
    return features['image'],features['label']


def input_fn(filepath, is_eval):
    print('FILE PATH:',filepath)
    batch_size = params.batch_size
    pixels = params.pixels
    classes = params.classes
    
    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(filepath)
        dataset = dataset.shard(hvd.size(),  hvd.rank())
        dataset = dataset.map(_parse_function, num_parallel_calls=hvd.local_size())
        dataset = dataset.shuffle(100* batch_size)
        if is_eval == False:
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size, drop_remainder=True)
        
    # Create your tf representation of the iterator
    iterator = dataset.make_one_shot_iterator()
    image,label = iterator.get_next()
    # Bring your picture back in shape
    image = tf.reshape(image, [-1,pixels, pixels,1])
    # Create a one hot array for your labels,1
    label= tf.one_hot(label, classes)
    

    # returns features x and labels y
    return _batch_normalization(image),label

def main(unused_argv):
    
    tf.logging.set_verbosity(tf.logging.INFO)
    
    # Get input data from directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, help='training data')

    args = parser.parse_args()

    input_data = os.path.join(args.input_data,'chest_xray')
    
    print("the input data is at %s" % input_data)
    
    
    # Horovod: initialize Horovod.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting them.
    model_dir = './cnn_model' if hvd.rank() == 0 else None

    # Create the Estimator
    cnn = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=model_dir,
      config=tf.estimator.RunConfig(session_config=config))

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=200)

    # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states from
    # rank 0 to all other processes. This is necessary to ensure consistent
    # initialization of all workers when training is started with random weights or
    # restored from a checkpoint.
    
    bcast_hook =hvd.BroadcastGlobalVariablesHook(0)

    # Train the model
    callable_train_input_fn = lambda: input_fn(os.path.join(input_data,'train.tfrecords'), False)
    #evaluate the model
    callable_eval_input_fn = lambda: input_fn(os.path.join(input_data,'test.tfrecords'), True)

    # Horovod: reduce number of training steps inversely proportional to the number
    # of workers.
    cnn.train(
      input_fn=callable_train_input_fn,
      steps=params.epochs // hvd.size(),
      hooks=[logging_hook,bcast_hook])


    if hvd.rank()==0 :
        eval_results = cnn.evaluate(input_fn=callable_eval_input_fn)
        print(eval_results)


if __name__ == "__main__":
    tf.app.run()
    