import math

import tensorflow as tf
from tensorflow.keras import layers

import horovod.tensorflow.keras as hvd

from azureml.core.run import Run

params = {
    'classes':10,
    'batch_size':128,
    'epochs':2,
    'num_batches':20,
    'width':28,
    'height':28,
    'channel':1,
}


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
    
    
    dataset = dataset.shard(hvd.size(),  hvd.rank())
    dataset = dataset.shuffle(100* params['batch_size'])
    if is_eval == False:
        dataset = dataset.repeat()
        
<<<<<<< HEAD
    '''iterator = dataset.make_one_shot_iterator()
    while True:
        image,label = iterator.get_next()
        yield image,label'''
=======
>>>>>>> 55d449e8120c2558602c742c82b87bbe13ca443f
    return dataset

def cnn_layers(input):
    x = layers.Conv2D(32, (3, 3),
                      activation='relu', padding='valid')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(params['classes'],
                               activation='softmax',
                               name='x_train_out')(x)
    return predictions


if __name__ == "__main__":
    
    tf.logging.set_verbosity(tf.logging.INFO)

    hvd.init()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    train_dataset = input_fn(X_train,y_train,is_eval=False)
    iterator = train_dataset.make_one_shot_iterator()
    
    inputs, targets = iterator.get_next()
    model_input = layers.Input(tensor=inputs)
    model_output = cnn_layers(model_input)
    model = tf.keras.models.Model(inputs=model_input, outputs=model_output)

    opt = tf.keras.optimizers.Adam(1e-3 * hvd.size())
    opt = hvd.DistributedOptimizer(opt)

    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]
    
    model.compile(opt,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    target_tensors=[targets])
    model.summary()
    
    if hvd.rank() == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                        './checkpoint-{epoch}.h5')
                        )
        
    steps_per_epoch = math.ceil(X_train.shape[0]//params['batch_size'])
    model.fit(epochs=params['epochs'],
                    steps_per_epoch=steps_per_epoch,
                    callbacks=callbacks)
   

    if hvd.rank()==0 :
        test_dataset = input_fn(X_test,y_test,is_eval=True)
        _, eval_acc = model.evaluate(test_dataset,steps=len(X_test))
        print('eval accuracy', eval_acc)

 