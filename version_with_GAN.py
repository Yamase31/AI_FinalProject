#imports
import copy
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
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
    
    with open('sudoku.csv') as csvfile:
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

def make_generator_model():
    Generator_model = keras.models.Sequential()
    Generator_model.add(layers.Dense(1 * 1 * 256, use_bias = False, input_shape = (9, 9, 1)))
    Generator_model.add(layers.BatchNormalization())
    Generator_model.add(layers.LeakyReLU())

    Generator_model.add(layers.Reshape((1, 1, 256)))
    assert Generator_model.output_shape == (None, 1, 1, 256)  # Note: None is the batch size

    Generator_model.add(layers.Conv2DTranspose(128, (5, 5), strides = (1, 1), padding = 'same', use_bias = False))
    assert Generator_model.output_shape == (None, 1, 1, 128)
    Generator_model.add(layers.BatchNormalization())
    Generator_model.add(layers.LeakyReLU())

    Generator_model.add(layers.Conv2DTranspose(64, (5, 5), strides = (3, 3), padding = 'same', use_bias = False))
    assert Generator_model.output_shape == (None, 3, 3, 64)
    Generator_model.add(layers.BatchNormalization())
    Generator_model.add(layers.LeakyReLU())

    Generator_model.add(layers.Conv2DTranspose(1, (5, 5), strides = (3, 3), padding = 'same', use_bias = False, activation = 'tanh'))
    print(Generator_model.output_shape)

    #output should be 81 x 9

    assert Generator_model.output_shape == (None, 9, 9, 1)
    return Generator_model

def make_discriminator_model():
    Discriminator_model = keras.models.Sequential()
    Discriminator_model.add(layers.Conv2D(64, kernel_size = (3,3), padding = 'same', input_shape = (9,9,1)))

    Discriminator_model.add(layers.LeakyReLU())
    Discriminator_model.add(layers.Dropout(0.3))

    Discriminator_model.add(layers.Conv2D(128, kernel_size = (1,1), padding = 'same'))
    Discriminator_model.add(layers.LeakyReLU())
    Discriminator_model.add(layers.Dropout(0.3))

    Discriminator_model.add(layers.Flatten())
    Discriminator_model.add(layers.Dense(81 * 9))

    return Discriminator_model

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def train_step(images):
    #?
    return pass

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)


def main():
    x_train, x_test, y_train, y_test = get_data('sudoku.csv')
    
    Generator_model = make_generator_model()
##    noise = tf.random.normal([9, 9])
##    generated_image = Generator_model(noise, training=False)

    Discriminator_model = make_discriminator_model()

    #define loss 
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 16

    # You will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

if __name__ == '__main__':
    main()

