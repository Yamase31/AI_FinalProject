#imports
import copy
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Activation
from keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Reshape
#added
import tensorflow as tf
import matplotlib.pyplot as plt


#### Need to implement the GAN here ------------------- \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/























#data preprocessing
def get_data(file): 

    data = pd.read_csv(file)

    feat_raw = data['quizzes']
    label_raw = data['solutions']

    feat = []
    label = []

    for i in feat_raw:
    
        x = np.array([int(j) for j in i]).reshape((9,9,1))
        feat.append(x)
    
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

#create the model
model = keras.models.Sequential()

model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', padding = 'same', input_shape = (9,9,1)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size = (1,1), activation = 'relu', padding = 'same'))

model.add(Flatten())
model.add(Dense(81 * 9))
model.add(Reshape((-1, 9)))
model.add(Activation('softmax'))

#compile the model with adam optimizer, sparse categorical crossentropy and accuracy metrics
adam = keras.optimizers.Adam(lr = .001)
model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = adam,
              metrics = ['accuracy'])

#visual of the model
model.summary()

#early stop training with a patience of 3
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor = "loss",
    min_delta = 0.001,
    patience = 3)

#fit the model with the data, batch size of 32, and _ epochs 
model.fit(x_train, y_train,
          batch_size = 32,
          epochs = 1000,
          callbacks = [early_stop])

#accuracy is calculated and printed
test_results = model.evaluate(x_test, y_test)
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












