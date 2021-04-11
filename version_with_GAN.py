#imports
import copy
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from keras.layers import Activation
from keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Reshape, LeakyReLU, Dropout
#added
import tensorflow as tf
import matplotlib.pyplot as plt
import csv

#probably going to need new imports
#some of the imports might be off in the way that they are called

#data pre-processing
def get_data(file): 
    
    feat_raw = []
    label_raw = []
    
    # Change directory to where YOUR saved CSV is
    
    with open(r'C:\Users\Will Medick\git\AI_FinalProject\sudoku.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader)
        for row in reader:
            feat_raw.append(row[0])
            label_raw.append(row[1])

    feat = []
    label = []

    for i in feat_raw:
        x = np.array([int(j) for j in i]).reshape((9,9,1))
        feat.append(x)
    
    # Normalize the puzzles between -1 and 1
    feat = np.array(feat)    
    feat = feat/9
    feat -= .5    
    
    for i in label_raw:
        x = np.array([int(j) for j in i]).reshape((81,1)) - 1
        
        label.append(x)   
    
    label = np.array(label)
    
    del(feat_raw)
    del(label_raw)

    x_train, x_test, y_train, y_test = train_test_split(feat, label, test_size = 0.2, random_state = 42)
    
    return x_train, x_test, y_train, y_test

#load the data
x_train, x_test, y_train, y_test = get_data('sudoku.csv')

#normalize data

#the generator

Generator_model = keras.models.Sequential()
Generator_model.add(tf.layers.Dense(7 * 7 * 256, use_bias = False, input_shape = (100,)))
Generator_model.add(layers.BatchNormalization())
Generator_model.add(layers.LeakyReLU())

Generator_model.add(layers.Reshape((7, 7, 256)))
assert Generator_model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

Generator_model.add(layers.Conv2DTranspose(128, (5, 5), strides = (1, 1), padding = 'same', use_bias = False))
assert Generator_model.output_shape == (None, 7, 7, 128)
Generator_model.add(layers.BatchNormalization())
Generator_model.add(layers.LeakyReLU())

Generator_model.add(layers.Conv2DTranspose(64, (5, 5), strides = (2, 2), padding = 'same', use_bias = False))
assert Generator_model.output_shape == (None, 14, 14, 64)
Generator_model.add(layers.BatchNormalization())
Generator_model.add(layers.LeakyReLU())

Generator_model.add(layers.Conv2DTranspose(1, (5, 5), strides = (2, 2), padding = 'same', use_bias = False, activation = 'tanh'))

#output should be 81 x 9
assert Generator_model.output_shape == (None, 81, 9, 1)

noise = tf.random.normal([1, 100])
generated_image = Generator_model(noise, training=False)






#the discriminator
Discriminator_model = keras.models.Sequential()
Discriminator_model.add(Conv2D(64, kernel_size = (3,3), padding = 'same', input_shape = (9,9,1)))

Discriminator_model.add(LeakyReLU())
Discriminator_model.add(Dropout(0.3))

Discriminator_model.add(Conv2D(128, kernel_size = (1,1), padding = 'same'))
Discriminator_model.add(LeakyReLU())
Discriminator_model.add(Dropout(0.3))

Discriminator_model.add(Flatten())
Discriminator_model.add(Dense(81 * 9))



#define loss 
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)



def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])







#\/\/\/\/\/\/\/\/\/\/ This is all from the old CNN







#compile the model with adam optimizer, sparse categorical crossentropy and accuracy metrics
adam = keras.optimizers.Adam(lr = .001)
CNN_model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = adam,
              metrics = ['accuracy'])

#visual of the model
CNN_model.summary()

#early stop training with a patience of 3
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor = "loss",
    min_delta = 0.001,
    patience = 3)

#fit the model with the data, batch size of 32, and _ epochs 
final_model = CNN_model.fit(x_train, y_train,
                            batch_size = 32,
                            epochs = 1000,
                            callbacks = [early_stop])

#accuracy is calculated and printed
test_results = CNN_model.evaluate(x_test, y_test)
print("Accuracy: ", test_results[1])

#Show graph of loss over time for training data
plt.plot(final_model.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training data'], loc = 'upper left')
plt.show()

#Show graph of accuracy over time for training data
plt.plot(final_model.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training data'], loc = 'upper left')
plt.show()






















