from math import ceil

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Conv2D,MaxPooling2D,
                                    Flatten,Dense, Dropout)

import horovod.tensorflow.keras as hvd

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


def input_fn(X,y, is_eval):
    
    def preprocess_fn(image, label):
        
        x = tf.reshape(tf.cast(image, tf.float32),(
                                    params['width'],
                                    params['height'], 
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

def main():
    
    tf.logging.set_verbosity(tf.logging.INFO)

    hvd.init()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    model = keras.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu')
              )
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(params['classes'], activation='softmax'))
    opt = keras.optimizers.Adam(1e-3 * hvd.size())

    opt = hvd.DistributedOptimizer(opt)

    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )

    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

    if hvd.rank() == 0:
        callbacks.append(keras.callbacks.ModelCheckpoint(
                        './checkpoint-{epoch}.h5')
                        )

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0],
                               params['width'],
                               params['height'], 
                               params['channel']
                             )

    fit = model.fit(input_fn(X_train,y_train,is_eval=False),
                    epochs=params['epochs'],
                    steps_per_epoch=ceil(60000/params['batch_size']),
                    callbacks=callbacks)


    if hvd.rank()==0 :
        _, eval_acc = model.evaluate(input_fn(
                                    X_test,
                                    y_test,
                                    is_eval=True)
                                    )
        print('eval accuracy', eval_acc)

if __name__ == "__main__":
    main()   