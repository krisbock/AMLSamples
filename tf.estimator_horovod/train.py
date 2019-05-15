import collections
import math
import os
import random

import numpy as np
#import pandas as pd
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import (Dense,Dropout,Conv2D,MaxPool2D,
                                    Reshape,Input,Flatten)
import horovod.tensorflow as hvd

from azureml.core.run import Run

class Parameters:
    def __init__(self):
        self.classes = 3
        self.batch_size = 64
        self.epochs = 10
        self.num_batches=20

params = Parameters()


def model_fn(features,labels,mode):
    
    print("I model_fn",features.get_shape())
    
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



def input_fn(dataset, is_eval):
    
    X,y = dataset
    tf.data.Dataset.from_tensor_slices((X,y))
    batch_size = params.batch_size
    classes = params.classes
    
    with tf.name_scope('input'):
        dataset = dataset.shard(hvd.size(),  hvd.rank())
        dataset = dataset.shuffle(100* batch_size)
        if is_eval == False:
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size, drop_remainder=True)
        
    iterator = dataset.make_one_shot_iterator()
    image,label = iterator.get_next()
    label= tf.one_hot(label, classes)
    
    return image,label

def main(unused_argv):
    
    tf.logging.set_verbosity(tf.logging.INFO)

    
    
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

    
    bcast_hook =hvd.BroadcastGlobalVariablesHook(0)

        
    
    mnist_train = tfds.load(name="mnist", split=tfds.Split.TRAIN)
    mnist_test = tfds.load(name="mnist", split=tfds.Split.TEST)
    # Train the model
    callable_train_input_fn = lambda: input_fn(mnist_train, False)
    #evaluate the model
    callable_eval_input_fn = lambda: input_fn(mnist_test, True)

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
    