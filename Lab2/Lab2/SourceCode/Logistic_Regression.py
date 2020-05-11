from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import TensorBoard
from time import time

dataset = pd.read_csv('heart.csv',index_col=0)
dataset.astype(float)

# Normalize values to range [0:1]
dataset /= dataset.max()

y = dataset['target']
X = dataset.drop(['target'], axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

np.random.seed(155)
model = Sequential() # create model
model.add(Dense(40, input_dim=12, activation='relu')) # hidden layer
model.add(Dense(20, input_dim=40, activation='relu'))
model.add(Dense(1, activation='softmax')) # output layer

model.compile(loss= keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.adamax(),
                  metrics=['accuracy'])


from keras import backend as K
# model = KerasClassifier(build_fn=createmodel,verbose=0)
#fit the model
tensorborad = TensorBoard(log_dir="logs/{}".format(time()))
history = model.fit(X_train, Y_train,batch_size=32,epochs=20,verbose=1,
           validation_data=(X_test, Y_test), callbacks=[tensorborad])

# make prediction
y_pred = model.predict_classes(X_test)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Loss:', score[0])
print('Accuracy:', score[1])
