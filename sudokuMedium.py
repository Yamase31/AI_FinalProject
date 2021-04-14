"""
https://medium.com/analytics-vidhya/demystifying-the-gan-using-a-1d-function-keras-bc7b861bb304
https://towardsdatascience.com/generative-adversarial-network-gan-for-dummies-a-step-by-step-tutorial-fdefff170391#:~:text=GAN%20Training&text=Step%201%20—%20Select%20a%20number,both%20fake%20and%20real%20images
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


#bump the size up later


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

    feat = []
    label = []
    
    for i in feat_raw:
        x = np.array([int(j) for j in i]).reshape((9,9))
        feat.append(x)
        
    # Normalize the puzzles between 0 and 1
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


"""
x_train, x_test, y_train, y_test = get_data('sudoku.csv')

sudoku_board_1 = y_train[0]

y_train_reshaped = sudoku_board_1.reshape([1,81])

#print("reshaped sudoku board: ", y_train_reshaped)

#print("reshaped sudoku board shape: ", y_train_reshaped.shape)

mapped = y_train_reshaped

mapped = mapped / 9

print("mapped: ", mapped)

print(mapped.shape)



def get_real_data(n):
    
##    # generate random numbers between range
##    x1 = np.random.uniform(low = 0, high = 1, size=n)
##    # x2 = np.cos(x1)  # use this for inverted u-shape function
##    x1 = x1.reshape((n, 1))
##    x2 = x1 * x1  # use this for u-shape function
##    x2 = x2.reshape((n, 1))
##    x3 = x2 * x2  # use this for u-shape function
##    x3 = x3.reshape((n, 1))

    samples = np.vstack((x1, x2, x3))
    
##    samples = np.array([[0.77777778, 0.22222222, 0.88888889, 0.11111111, 0.66666667, 0.33333333,
## 0.55555556, 0.44444444, 0.,         0.33333333, 0.66666667, 0.44444444,
## 0.,         0.55555556, 0.88888889, 0.11111111, 0.77777778, 0.22222222,
## 0.55555556, 0.,         0.11111111, 0.44444444, 0.22222222, 0.77777778,
## 0.66666667, 0.88888889, 0.33333333, 0.        , 0.44444444, 0.66666667,
## 0.33333333, 0.77777778, 0.22222222, 0.88888889, 0.11111111, 0.55555556,
## 0.22222222, 0.88888889, 0.33333333, 0.55555556, 0.11111111, 0.44444444,
## 0.        , 0.66666667, 0.77777778, 0.11111111, 0.55555556, 0.77777778,
## 0.88888889, 0.        , 0.66666667, 0.22222222, 0.33333333, 0.44444444,
## 0.44444444, 0.77777778, 0.        , 0.22222222, 0.88888889, 0.11111111,
## 0.33333333, 0.55555556, 0.66666667, 0.66666667, 0.11111111, 0.55555556,
## 0.77777778, 0.33333333, 0.        , 0.44444444, 0.22222222, 0.88888889,
## 0.88888889, 0.33333333, 0.22222222, 0.66666667, 0.44444444, 0.55555556,
## 0.77777778, 0.,         0.11111111,]])
    return samples

# get 100 data points
data = get_real_data(2)

print("data: ", data)
print(data.shape)
"""


x_train, x_test, y_train, y_test = get_data('sudoku.csv')

def get_real_data(n, y_train):
    sudoku_board_0 = y_train[0]
    sudoku_board_reshaped_0 = sudoku_board_0.reshape([1,81])
    mapped_sudoku_board_0 = sudoku_board_reshaped_0
    mapped_sudoku_board_0 = mapped_sudoku_board_0 / 9

    sudoku_board_1 = y_train[1]
    sudoku_board_reshaped_1 = sudoku_board_1.reshape([1,81])
    mapped_sudoku_board_1 = sudoku_board_reshaped_1
    mapped_sudoku_board_1 = mapped_sudoku_board_1 / 9

    sudoku_board_2 = y_train[2]
    sudoku_board_reshaped_2 = sudoku_board_2.reshape([1,81])
    mapped_sudoku_board_2 = sudoku_board_reshaped_2
    mapped_sudoku_board_2 = mapped_sudoku_board_2 / 9

    #print("mapped: ", mapped_sudoku_board)
    #print("mapped shape: ", mapped_sudoku_board.shape)

    data1 = np.vstack((mapped_sudoku_board_0, mapped_sudoku_board_1, mapped_sudoku_board_2))
        
    return data1

# get 100 data points
data = get_real_data(3, y_train)

print("data", data)
print("data shape", data.shape)


















# convert to float for dl model
data = data.astype('float32')
#plt.scatter(data[:,0], data[:, 1], data[:, 2])

# convert your data into tensorflow data type.
train_data = tf.data.Dataset.from_tensor_slices(data)
train_data = train_data.batch(64).prefetch(32)

# discriminator model
def build_discriminator(n=81): #changes - based on the number of columns
  inps = layers.Input(shape=(n,))
  x = layers.Dense(25, activation='relu',
                   kernel_initializer='he_uniform')(inps)
  outs = layers.Dense(1)(x)
  model = keras.Model(inps, outs, name='discriminator')
  return model
# generator model
def build_generator(latent_dim=5):
  inps = layers.Input(shape=(latent_dim,))
  x = layers.Dense(81, activation='relu', #changes - based on the number of columns
                   kernel_initializer='he_uniform')(inps)
  outs = layers.Dense(81, activation='tanh')(x) #changes - based on the number of columns
  model = keras.Model(inps, outs, name='generator')
  return model
discriminator = build_discriminator()
generator = build_generator()
#print(discriminator.summary(), generator.summary())

class GAN(keras.Model):
    
  # initialize models with latent dimensions
  def __init__(self, disc, gen, latent_dim=5):
    super(GAN, self).__init__()
    self.discriminator = disc
    self.generator = gen
    self.latent_dim = latent_dim
  
  # compile with optimizers and loss function
  def compile(self, optD, optG, loss_fn):
    super(GAN, self).compile()
    self.optD = optD
    self.optG = optG
    self.loss_fn = loss_fn
    
  # custom training function
  def train_step(self, real_data):
    if isinstance(real_data, tuple):
      real_data = real_data[0]
    
    # get current batch size
    bs = tf.shape(real_data)[0]
    z = tf.random.normal(shape=(bs, self.latent_dim))
    fake_data = self.generator(z)
    
    # combine real and fake images in a single vector along with their labels
    combined_data = tf.concat([real_data, fake_data], axis=0)
    labels = tf.concat([tf.ones((bs, 1)), tf.zeros((bs, 1))], axis=0)
    
    # train your discriminator
    with tf.GradientTape() as tape:
      preds = self.discriminator(combined_data)
      d_loss = self.loss_fn(labels, preds)
    grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
    self.optD.apply_gradients(zip(grads, self.discriminator.trainable_weights))
    
    # misleading labels for generator
    misleading_labels = tf.ones((bs, 1))
    z = tf.random.normal(shape=(bs, self.latent_dim))
    
    # train your generator
    with tf.GradientTape() as tape:
      fake_preds = self.discriminator(self.generator(z))
      g_loss = self.loss_fn(misleading_labels, fake_preds)
    grads = tape.gradient(g_loss, self.generator.trainable_weights)
    self.optG.apply_gradients(zip(grads, self.generator.trainable_weights))
    return {"d_loss": d_loss, "g_loss": g_loss}
# create GAN model using already built D and G
gan = GAN(discriminator, generator)
# compile your model with loss and optimizers
gan.compile(
    keras.optimizers.Adam(),
    keras.optimizers.Adam(),
    keras.losses.BinaryCrossentropy(from_logits=True)
)

def show_samples(epoch, generator, data, n = 100, l_dim=5):
  # save results after every 20 epochs  
  if epoch % 20 == 0:
    z = tf.random.normal(shape=(n, l_dim))
    generated_data = generator(z)
    generated_points_list.append(generated_data)

# list for storing generated points
generated_points_list = []

# a lambda callback
cbk = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch,logs: show_samples(epoch, gan.generator, data))

hist = gan.fit(train_data, epochs=5, callbacks=[cbk], verbose=True)
'''
this will almost take 40-50 seconds but you can turn on the verbose and see progress along the way
'''

"""
# plot the results
plt.plot(hist.history['d_loss'], color='blue', label='discriminator loss')
plt.plot(hist.history['g_loss'], color='red', label='generator loss')
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.legend()
plt.show()
"""



print("first data point: ", data[0])

print("first data point shape: ", data[0].shape)

print("\n")
print("\n")
print("\n")
print("\n")

"""
#to print all 100 new sudoku puzzles
count = 0
for i in range(len(generated_points_list)):
    for j in range(len(generated_points_list[i])):
        print(generated_points_list[i][j], end=' ')
        count += 1
    print()
print(count)
"""

print("generated puzzle: ", generated_points_list[0][0])

print("generated puzzle shape: ", generated_points_list[0].shape)

un_normalized_generated_puzzle = np.reshape(generated_points_list[0][0], (9, 9)).T

#take the absolute value of all of the puzzle values
un_normalized_generated_puzzle = np.absolute(un_normalized_generated_puzzle)

#un normalize the values by multiplying by 9
un_normalized_generated_puzzle = un_normalized_generated_puzzle * 9

#round all of the values
un_normalized_generated_puzzle = np.round(un_normalized_generated_puzzle, 0)

print("new puzzle: ", un_normalized_generated_puzzle)

print("new puzzle shape: ", un_normalized_generated_puzzle.shape) #should be (9,9)

#error correction mechanism for the new outputted puzzles:





"""
# get real data to show with fake data
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
# anim.save('animation.gif', fps=2)
anim.save('animation.mp4', fps=10)
"""
