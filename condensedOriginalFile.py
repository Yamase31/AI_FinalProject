#imports
import copy
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Activation
from keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Reshape

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
model.add(Dense(81*9))
model.add(Reshape((-1, 9)))
model.add(Activation('softmax'))

#train the model
adam = keras.optimizers.Adam(lr = .001)
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = adam)

model.fit(x_train, y_train, batch_size = 32, epochs = 2)

#solve sudoku by filling blank positions one by one
def norm(a):
    
    return (a/9)-.5

def denorm(a):
    
    return (a+.5)*9

def inference_sudoku(sample):
    
    '''
        This function solve the sudoku by filling blank positions one by one.
    '''
    
    feat = copy.copy(sample)
    
    while(1):
    
        out = model.predict(feat.reshape((1,9,9,1)))  
        out = out.squeeze()

        pred = np.argmax(out, axis = 1).reshape((9,9)) + 1
        prob = np.around(np.max(out, axis = 1).reshape((9,9)), 2) 
        
        feat = denorm(feat).reshape((9,9))
        mask = (feat==0)
     
        if(mask.sum()==0):
            break
            
        prob_new = prob * mask
    
        ind = np.argmax(prob_new)
        x, y = (ind//9), (ind%9)

        val = pred[x][y]
        feat[x][y] = val
        feat = norm(feat)
    
    return pred

#testing 100 games
def test_accuracy(feats, labels):
    
    correct = 0
    
    for i,feat in enumerate(feats):
        
        pred = inference_sudoku(feat)
        
        true = labels[i].reshape((9,9))+1
        
        if(abs(true - pred).sum()==0):
            correct += 1
        
    print(correct/feats.shape[0])

test_accuracy(x_test[:100], y_test[:100])

#test our own game
def solve_sudoku(game):
    
    game = game.replace('\n', '')
    game = game.replace(' ', '')
    game = np.array([int(j) for j in game]).reshape((9,9,1))
    game = norm(game)
    game = inference_sudoku(game)
    return game

game = '''
          0 8 0 0 3 2 0 0 1
          7 0 3 0 8 0 0 0 2
          5 0 0 0 0 7 0 3 0
          0 5 0 0 0 1 9 7 0
          6 0 0 7 0 9 0 0 8
          0 4 7 2 0 0 0 5 0
          0 2 0 6 0 0 0 0 9
          8 0 0 0 9 0 3 0 5
          3 0 0 8 2 0 0 1 0
      '''

game = solve_sudoku(game)

print('solved puzzle:\n')
print(game)

np.sum(game, axis = 1)








