import tensorflow as tf
import numpy as np
from tensorflow import keras

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.4:
            print('\nLoss is low so cancelling training!')
            self.model.stop_training = True
callbacks = myCallback()
fmnist = tf.keras.datasets.fashion_mnist



(training_images, training_lables),(test_images, test_labels) = fmnist.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0
training_images = training_images.reshape(-1,28,28,1)
test_images = test_images.reshape(-1,28,28,1)
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = (28, 28, 1)),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(10, activation='softmax')])
print(model.summary())

model.compile(optimizer ='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(training_images,training_lables, epochs=15, callbacks=[callbacks])
print(model.evaluate(test_images, test_labels))




'''

#this over here is a simple 1 nueron nueral network to accomplish a simple task of figuring out a pattern

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
model.fit(xs, ys, epochs=600)
print(model.predict([10.0]))
'''

'''
fmnist = tf.keras.datasets.fashion_mnist

(training_images, training_lables),(test_images, test_labels) = fmnist.load_data()

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(training_images,training_lables, epochs=15, callbacks=[callbacks])
print(model.evaluate(test_images, test_labels))
'''