# CSCI 315: Artificial Intelligence - Final Project
## Solving sudoku puzzles using a CNN (Convolutional Neural Network)

### Authors: Laurie Jones, Jacob Flood, Will Medick, James Lawson, and Evan Phaup

Orginial project that we based our project off of: https://github.com/shivaverma/Sudoku-Solver

Article that pairs with the original project: https://towardsdatascience.com/solving-sudoku-with-convolution-neural-network-keras-655ba4be3b11

Since our data is larger than 25MB, we have attached it here: https://drive.google.com/file/d/1nFx33BdZe8ydhUFNrsaMeqFU2uDdO28f/view?usp=sharing - original source of out data: https://www.kaggle.com/bryanpark/sudoku?select=sudoku.csv

Epoch 1/1000 25000/25000 [==============================] - 1887s 75ms/step - loss: 0.6441 - accuracy: 0.7298
Epoch 2/1000 25000/25000 [==============================] - 1887s 75ms/step - loss: 0.3606 - accuracy: 0.8268
Epoch 3/1000 25000/25000 [==============================] - 1678s 67ms/step - loss: 0.3500 - accuracy: 0.8342
Epoch 4/1000 25000/25000 [==============================] - 1685s 67ms/step - loss: 0.3433 - accuracy: 0.8387
Epoch 5/1000 25000/25000 [==============================] - 1730s 69ms/step - loss: 0.3389 - accuracy: 0.8417
Epoch 6/1000 25000/25000 [==============================] - 1727s 69ms/step - loss: 0.3360 - accuracy: 0.8436
Epoch 7/1000 25000/25000 [==============================] - 1704s 68ms/step - loss: 0.3336 - accuracy: 0.8452
Epoch 8/1000 25000/25000 [==============================] - 1671s 67ms/step - loss: 0.3319 - accuracy: 0.8463
Epoch 9/1000 25000/25000 [==============================] - 1668s 67ms/step - loss: 0.3305 - accuracy: 0.8472
Epoch 10/1000 25000/25000 [==============================] - 1809s 72ms/step - loss: 0.3292 - accuracy: 0.8480
Epoch 11/1000 25000/25000 [==============================] - 2127s 85ms/step - loss: 0.3280 - accuracy: 0.8488
Epoch 12/1000 25000/25000 [==============================] - 2069s 83ms/step - loss: 0.3272 - accuracy: 0.8493
Epoch 13/1000 25000/25000 [==============================] - 2039s 82ms/step - loss: 0.3263 - accuracy: 0.8499
Epoch 14/1000 25000/25000 [==============================] - 1773s 71ms/step - loss: 0.3256 - accuracy: 0.8503
Epoch 15/1000 25000/25000 [==============================] - 1673s 67ms/step - loss: 0.3245 - accuracy: 0.8510
Epoch 16/1000 25000/25000 [==============================] - 1674s 67ms/step - loss: 0.3238 - accuracy: 0.8515
Epoch 17/1000 25000/25000 [==============================] - 1806s 72ms/step - loss: 0.3233 - accuracy: 0.8518
Epoch 18/1000 25000/25000 [==============================] - 2160s 86ms/step - loss: 0.3226 - accuracy: 0.8522
Epoch 19/1000 25000/25000 [==============================] - 2540s 102ms/step - loss: 0.3223 - accuracy: 0.8525
Epoch 20/1000 25000/25000 [==============================] - 1770s 71ms/step - loss: 0.3218 - accuracy: 0.8528
Epoch 21/1000 25000/25000 [==============================] - 1573s 63ms/step - loss: 0.3214 - accuracy: 0.8530
Epoch 22/1000 25000/25000 [==============================] - 1601s 64ms/step - loss: 0.3211 - accuracy: 0.8532
Epoch 23/1000 25000/25000 [==============================] - 1583s 63ms/step - loss: 0.3207 - accuracy: 0.8534
Epoch 24/1000 25000/25000 [==============================] - 1584s 63ms/step - loss: 0.3205 - accuracy: 0.8536
Epoch 25/1000 25000/25000 [==============================] - 1517s 61ms/step - loss: 0.3202 - accuracy: 0.8537
Epoch 26/1000 25000/25000 [==============================] - 1543s 62ms/step - loss: 0.3197 - accuracy: 0.8540
Epoch 27/1000 25000/25000 [==============================] - 1559s 62ms/step - loss: 0.3197 - accuracy: 0.8541
Epoch 28/1000 25000/25000 [==============================] - 1589s 64ms/step - loss: 0.3193 - accuracy: 0.8543
Epoch 29/1000 25000/25000 [==============================] - 1602s 64ms/step - loss: 0.3190 - accuracy: 0.8544
Epoch 30/1000 25000/25000 [==============================] - 1586s 63ms/step - loss: 0.3189 - accuracy: 0.8546
Epoch 31/1000 25000/25000 [==============================] - 1603s 64ms/step - loss: 0.3187 - accuracy: 0.8547
Epoch 32/1000 25000/25000 [==============================] - 1609s 64ms/step - loss: 0.3183 - accuracy: 0.8549
Epoch 33/1000 25000/25000 [==============================] - 1609s 64ms/step - loss: 0.3183 - accuracy: 0.8549
Epoch 34/1000 25000/25000 [==============================] - 1602s 64ms/step - loss: 0.3180 - accuracy: 0.8552
Epoch 35/1000 25000/25000 [==============================] - 1597s 64ms/step - loss: 0.3179 - accuracy: 0.8552
Epoch 36/1000 25000/25000 [==============================] - 1594s 64ms/step - loss: 0.3176 - accuracy: 0.8553
Epoch 37/1000 25000/25000 [==============================] - 1599s 64ms/step - loss: 0.3174 - accuracy: 0.8554
Epoch 38/1000 25000/25000 [==============================] - 1593s 64ms/step - loss: 0.3173 - accuracy: 0.8555
Epoch 39/1000 25000/25000 [==============================] - 1594s 64ms/step - loss: 0.3170 - accuracy: 0.8557
Epoch 40/1000 25000/25000 [==============================] - 1602s 64ms/step - loss: 0.3170 - accuracy: 0.8557
Epoch 41/1000 25000/25000 [==============================] - 1592s 64ms/step - loss: 0.3168 - accuracy: 0.8558
Epoch 42/1000 25000/25000 [==============================] - 1602s 64ms/step - loss: 0.3166 - accuracy: 0.8559
Epoch 43/1000 25000/25000 [==============================] - 1648s 66ms/step - loss: 0.3164 - accuracy: 0.8560
Epoch 44/1000 25000/25000 [==============================] - 2261s 90ms/step - loss: 0.3164 - accuracy: 0.8561
Epoch 45/1000 25000/25000 [==============================] - 1936s 77ms/step - loss: 0.3162 - accuracy: 0.8562
Epoch 46/1000 25000/25000 [==============================] - 1638s 66ms/step - loss: 0.3161 - accuracy: 0.8563
Epoch 47/1000 25000/25000 [==============================] - 1636s 65ms/step - loss: 0.3160 - accuracy: 0.8563
Epoch 48/1000 25000/25000 [==============================] - 1586s 63ms/step - loss: 0.3157 - accuracy: 0.8565
Epoch 49/1000 25000/25000 [==============================] - 1631s 65ms/step - loss: 0.3156 - accuracy: 0.8565
Epoch 50/1000 25000/25000 [==============================] - 1825s 73ms/step - loss: 0.3154 - accuracy: 0.8566
Epoch 51/1000 25000/25000 [==============================] - 2662s 106ms/step - loss: 0.3154 - accuracy: 0.8566
Epoch 52/1000 25000/25000 [==============================] - 2819s 113ms/step - loss: 0.3152 - accuracy: 0.8568
Epoch 53/1000 25000/25000 [==============================] - 2568s 103ms/step - loss: 0.3152 - accuracy: 0.8568
Epoch 54/1000 25000/25000 [==============================] - 2536s 101ms/step - loss: 0.3151 - accuracy: 0.8568
Epoch 55/1000 25000/25000 [==============================] - 2622s 105ms/step - loss: 0.3149 - accuracy: 0.8570
Epoch 56/1000 25000/25000 [==============================] - 3122s 125ms/step - loss: 0.3148 - accuracy: 0.8570
Epoch 57/1000 25000/25000 [==============================] - 2924s 117ms/step - loss: 0.3146 - accuracy: 0.8572
Epoch 58/1000 25000/25000 [==============================] - 2680s 107ms/step - loss: 0.3145 - accuracy: 0.8572
Epoch 59/1000 25000/25000 [==============================] - 3051s 122ms/step - loss: 0.3145 - accuracy: 0.8572
