"""
https://medium.com/analytics-vidhya/demystifying-the-gan-using-a-1d-function-keras-bc7b861bb304
https://towardsdatascience.com/generative-adversarial-network-gan-for-dummies-a-step-by-step-tutorial-fdefff170391#:~:text=GAN%20Training&text=Step%201%20—%20Select%20a%20number,both%20fake%20and%20real%20images
https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback
"""

#imports
import csv
import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Conv1D, Conv1DTranspose
from keras.layers import BatchNormalization, Input, Dense
import time
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
from functools import reduce

#data pre-processing
def get_data(file):
    
    feat_raw = []
    label_raw = []
    
    with open('sudoku.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader)
        for row in reader:
            feat_raw.append(row[0])
            label_raw.append(row[1])

    feat = []
    label = []
    
    for i in feat_raw:
        x = np.array([int(j) for j in i]).reshape((9,9))
        feat.append(x)
        
    #puzzles are normalized
    feat = np.array(feat)    
    feat = feat/9
    feat -= .5
    
    for i in label_raw:
        x = np.array([int(j) for j in i]).reshape((81,1)) - 1
        
        label.append(x)   

    #converted to a numpy array
    label = np.array(label)

    #un-needed lists are deleted
    del(feat_raw)
    del(label_raw)

    #train test split function is called
    x_train, x_test, y_train, y_test = train_test_split(feat, label, test_size = 0.2, random_state = 42)

    #data is returned
    return x_train, x_test, y_train, y_test

#get all of our data
x_train, x_test, y_train, y_test = get_data('sudoku.csv')

#get our data from y_train
def get_real_data(y_train):
    #specify how many boards we want to import 800000
    sudoku_boards = y_train[0:799999]
    #the number on the left needs to be the number of boards, reshape
    sudoku_boards = sudoku_boards.reshape([799999,81])
    #divide by 9 to convert the boards from 81 integers to 9x9
    sudoku_boards = sudoku_boards / 9 

    #boards are returned
    return sudoku_boards

#get our data points
data = get_real_data(y_train)

#convert to float for dl model
data = data.astype('float32')

#convert your data into tensorflow data type.
train_data = tf.data.Dataset.from_tensor_slices(data)
train_data = train_data.batch(64).prefetch(32)

#discriminator model
def build_discriminator(n=81): #changes - based on the number of columns
  inps = layers.Input(shape=(n,))
  x = layers.Dense(25, activation='relu',
                   kernel_initializer='he_uniform')(inps)
  outs = layers.Dense(1)(x)
  model = keras.Model(inps, outs, name='discriminator')
  return model

#generator model
def build_generator(latent_dim=5):
  inps = layers.Input(shape=(latent_dim,))
  x = layers.Dense(81, activation='relu', #changes - based on the number of columns
                   kernel_initializer='he_uniform')(inps)
  outs = layers.Dense(81, activation='tanh')(x) #changes - based on the number of columns
  model = keras.Model(inps, outs, name='generator')
  return model

#builds our discriminator
discriminator = build_discriminator()

#builds our generator
generator = build_generator()

#model summaries are primted
print(discriminator.summary(), generator.summary())

class GAN(keras.Model):
    
  #initialize models with latent dimensions
  def __init__(self, disc, gen, latent_dim=5):
    super(GAN, self).__init__()
    self.discriminator = disc
    self.generator = gen
    self.latent_dim = latent_dim
  
  #compile with optimizers and loss function
  def compile(self, optD, optG, loss_fn):
    super(GAN, self).compile()
    self.optD = optD
    self.optG = optG
    self.loss_fn = loss_fn
    
  #custom training function
  def train_step(self, real_data):
    if isinstance(real_data, tuple):
      real_data = real_data[0]
    
    #get current batch size
    bs = tf.shape(real_data)[0]
    z = tf.random.normal(shape=(bs, self.latent_dim))
    fake_data = self.generator(z)
    
    #combine real and fake images in a single vector along with their labels
    combined_data = tf.concat([real_data, fake_data], axis=0)
    labels = tf.concat([tf.ones((bs, 1)), tf.zeros((bs, 1))], axis=0)
    
    #train your discriminator
    with tf.GradientTape() as tape:
      preds = self.discriminator(combined_data)
      d_loss = self.loss_fn(labels, preds)
    grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
    self.optD.apply_gradients(zip(grads, self.discriminator.trainable_weights))
    
    #misleading labels for generator
    misleading_labels = tf.ones((bs, 1))
    z = tf.random.normal(shape=(bs, self.latent_dim))
    
    #train the generator
    with tf.GradientTape() as tape:
      fake_preds = self.discriminator(self.generator(z))
      g_loss = self.loss_fn(misleading_labels, fake_preds)
    grads = tape.gradient(g_loss, self.generator.trainable_weights)
    self.optG.apply_gradients(zip(grads, self.generator.trainable_weights))
    return {"d_loss": d_loss, "g_loss": g_loss}

#create GAN model using already built D and G
gan = GAN(discriminator, generator)

#compile your model with loss and optimizers
gan.compile(
    keras.optimizers.Adam(),
    keras.optimizers.Adam(),
    keras.losses.BinaryCrossentropy(from_logits=True)
)

def show_samples(epoch, generator, data, n = 100, l_dim=5):
  #save results after every 20 epochs  
  if epoch % 20 == 0:
    z = tf.random.normal(shape=(n, l_dim))
    generated_data = generator(z)
    generated_points_list.append(generated_data)

#list for storing generated points
generated_points_list = []

#callback
cbk = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch,logs: show_samples(epoch, gan.generator, data))

#fit the GAN
hist = gan.fit(train_data, epochs=20, callbacks=[cbk], verbose=True)

#plot the results
plt.plot(hist.history['d_loss'], color='blue', label='discriminator loss')
plt.plot(hist.history['g_loss'], color='red', label='generator loss')
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.legend()
plt.show()

#print the generated puzzle that is still normalized
print("generated puzzle normalized: ", generated_points_list[0][0])

#transpose our puzzle into a 9x9 shape
un_normalized_generated_puzzle = np.reshape(generated_points_list[0][0], (9, 9)).T

#take the absolute value of all of the puzzle values
un_normalized_generated_puzzle = np.absolute(un_normalized_generated_puzzle)

#un normalize the values by multiplying by 9
un_normalized_generated_puzzle = un_normalized_generated_puzzle * 9

#round all of the values
un_normalized_generated_puzzle = np.round(un_normalized_generated_puzzle, 0)

#print the puzzle
print("new puzzle: ", un_normalized_generated_puzzle)

#difficulty setting (default is 7, deletes all values lower than 7) ----------------------------
#this is a very hard difficulty setting
difficulty_setting = 7

#remove all of the values less than the difficulty level
un_normalized_generated_puzzle[un_normalized_generated_puzzle<7] = 0

#print the puzzle with difficulty
print("puzzle with difficulty setting: ", un_normalized_generated_puzzle)

#create a GIF
real_x, real_y  = data[:, 0], data[:, 1]
camera = Camera(plt.figure())
plt.xlim(real_x.min()-0.2, real_x.max()+0.2)
plt.ylim(real_y.min()-0.05, real_y.max()+0.05)
for i in range(len(generated_points_list)):
  plt.scatter(real_x, real_y, color='blue')
  fake_x, fake_y = generated_points_list[i][:, 0], generated_points_list[i][:, 1]
  plt.scatter(fake_x, fake_y, color='red')
  camera.snap()
anim = camera.animate(blit=True)
plt.close()
anim.save('animation.gif', fps=2)


