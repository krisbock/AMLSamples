import tensorflow as tf
from tensorflow.keras.layers import (Dense,Dropout,Conv2D,MaxPool2D,
                                    Reshape,Input,Flatten)
import horovod.tensorflow as hvd
from azureml.core.run import Run

params = {
    'classes':10,
    'batch_size':128,
    'epochs':200,
    'num_batches':20,
    'width':28,
    'height':28,
    'channel':1,
}

def model_fn(features,labels,mode):
    
    input_shape = tf.shape(features)
    input_layer = Input(tensor=features,shape=input_shape)

    conv1 = Conv2D(32,(5,5),activation=tf.nn.relu)(input_layer)
    max_pool = MaxPool2D(pool_size=(2,2))(conv1)
    conv2 = Conv2D(64,(5,5),activation=tf.nn.relu)(max_pool)
    max_pool2 = MaxPool2D(pool_size=(2,2))(conv2)
    flatten = Flatten(input_shape=(-1, params['height'],
                                   params['width'], 
                                   params['channel']))(max_pool2)
    full_connected = Dense(1024)(flatten)
    drop = Dropout(.4)(full_connected)
    logits = Dense(params['classes'])(drop)

    predictions = {"classes":tf.argmax(input=logits, axis=1),
                  "probabilities":tf.nn.softmax(logits, name="softmax_tensor")}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)
     

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32),
                               depth=params['classes'])
    
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                           logits=logits)
    run = Run.get_context()
    
   
    if mode == tf.estimator.ModeKeys.TRAIN:
        lr = 1e-2
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr * hvd.size(),
                                               momentum=0.9
                                              )
        optimizer = hvd.DistributedOptimizer(optimizer)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step()
                                     )
        
        return tf.estimator.EstimatorSpec(mode=mode, 
                                          loss=loss, 
                                          train_op=train_op
                                         )

    
    eval_metric_ops = {"Accuracy": tf.metrics.accuracy(labels=tf.argmax(labels,axis=1), 
                                                        predictions=predictions["classes"])
                      }  
    run.log('accuracy',eval_metric_ops["Accuracy"][1])
  

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def input_fn(X,y, is_eval):
    
    def preprocess_fn(image, label):
        
        x = tf.reshape(tf.cast(image, tf.float32),(
                                    params['height'],
                                    params['width'], 
                                    params['channel'])
                      )
        x = tf.cast(x, tf.float32)/255
        y = tf.one_hot(tf.cast(label, tf.uint8), params['classes'])
        return x, y
    
    dataset = tf.data.Dataset.from_tensor_slices((X,y))
    dataset = dataset.apply(tf.contrib.data.map_and_batch(preprocess_fn,
                            params['batch_size']))
    
    
    with tf.name_scope('input'):
        dataset = dataset.shard(hvd.size(),  hvd.rank())
        dataset = dataset.shuffle(100* params['batch_size'])
        if is_eval == False:
            dataset = dataset.repeat()
        
    iterator = dataset.make_one_shot_iterator()
    image,label = iterator.get_next()


    return image,label

def main(unused_argv):
    
    tf.logging.set_verbosity(tf.logging.INFO)

    hvd.init()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    model_dir = './cnn_model' if hvd.rank() == 0 else None

    cnn = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=model_dir,
      config=tf.estimator.RunConfig(session_config=config))

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, 
                                              every_n_iter=20,
                                              at_end = True)    
    bcast_hook =hvd.BroadcastGlobalVariablesHook(0)

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    callable_train_input_fn = lambda: input_fn(X_train, y_train, False)
    callable_eval_input_fn = lambda: input_fn(X_test, y_test, True)
    
    cnn.train(
      input_fn=callable_train_input_fn,
      steps=params['epochs'] // hvd.size(),
      hooks=[logging_hook,bcast_hook])


    if hvd.rank()==0 :
        eval_results = cnn.evaluate(input_fn=callable_eval_input_fn)
        print('eval accuracy', eval_results)


if __name__ == "__main__":
    tf.app.run()
    