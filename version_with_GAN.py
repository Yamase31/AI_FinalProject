
# Libraries we are currently using

import csv
import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Conv2D
import time

#Libraries we are not using currently / probably don't need

# import pandas as pd
# import copy
# import glob
# import imageio
# import os
# import PIL
# from IPython import display


# Make sure the optimizer variables are global
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Data pre-processing

def get_data(file): 
    
    feat_raw = []
    label_raw = []
    
    with open('sudoku.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for i in range(999900):
            next(reader)
        for row in reader:
            feat_raw.append(row[0])
            label_raw.append(row[1])
    print(feat_raw)       
    feat = []
    label = []
    
    for i in feat_raw:
        x = np.array([int(j) for j in i]).reshape((9,9))
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

def make_generator_model():
    
    inps = layers.Input(shape=(9,9))
    x = layers.Dense(9, activation='relu',
        kernel_initializer='he_uniform')(inps)
    # x = Conv2D(64, kernel_size = (3,3),
        # activation = 'relu', padding = 'same',
        # input_shape = (9,9))(inps)
    outs = layers.Dense(9, activation='tanh')(x)
    model = keras.Model(inps, outs, name='generator')
    
    # old model
    
    # model = keras.Sequential()
    # model.add(layers.Dense(1 * 1 * 256, use_bias = False))#, input_shape = (9, 9, 1)))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())
    #
    # model.add(layers.Reshape((1, 1, 256)))
# #    assert model.output_shape == (None, 1, 1, 256)  # Note: None is the batch size
#
    # model.add(layers.Conv2DTranspose(128, (5, 5), strides = (1, 1), padding = 'same', use_bias = False))
# #   assert model.output_shape == (None, 1, 1, 128)
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())
    #
    # model.add(layers.Conv2DTranspose(64, (5, 5), strides = (3, 3), padding = 'same', use_bias = False))
# #    assert model.output_shape == (None, 3, 3, 64)
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())
    #
    # model.add(layers.Conv2DTranspose(1, (5, 5), strides = (3, 3), padding = 'same', use_bias = False, activation = 'tanh'))
# #    assert model.output_shape == (None, 9, 9, 1)
    
    return model

def make_discriminator_model():
    
    inps = layers.Input(shape=(9,9))
    x = layers.Dense(9, activation='relu',
        kernel_initializer='he_uniform')(inps)
    outs = layers.Dense(1)(x)
    model = keras.Model(inps, outs, name='discriminator')
    
    # old model
    
    # model = keras.Sequential()
    # model.add(layers.Conv2D(64, kernel_size = (3,3), padding = 'same'))#, input_shape = (9,9,1)))
    #
    # model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))
    #
    # model.add(layers.Conv2D(128, kernel_size = (1,1), padding = 'same'))
    # model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))
    #
    # model.add(layers.Flatten())
    # model.add(layers.Dense(1))

    return model


def discriminator_loss(real_output, fake_output):
    #define loss 
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    #define loss 
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".

@tf.function
def train_step(images, generator, discriminator,noise_dim):
##    noise = tf.random.uniform([9, 9], 0, 9, dtype=tf.dtypes.int64)
    noise = tf.random.normal([9, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs, generator, discriminator, noise_dim, seed):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch, generator, discriminator, noise_dim)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

def main():
    
    x_train, x_test, y_train, y_test = get_data('sudoku.csv')
    BUFFER_SIZE = 800000
    BATCH_SIZE = 800000
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    noise = tf.random.uniform([9, 9], 0, 9, dtype=tf.dtypes.int64)
    print(noise)
    generated_image = generator(noise, training = False) #either
    print(generator.summary(), discriminator.summary())

    EPOCHS = 50
    noise_dim = 9
    num_examples_to_generate = 16

    # You will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    train(train_dataset, EPOCHS, generator, discriminator, noise_dim, seed)

if __name__ == '__main__':
    main()
