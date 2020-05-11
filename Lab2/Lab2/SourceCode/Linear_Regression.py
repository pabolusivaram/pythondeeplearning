from keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler
import keras
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras import models, layers
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from time import time

# Load the dataset
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

# let us view on sample from the features
print(X_train[0], y_train[0])

scaler = StandardScaler()

# first we fit the scaler on the training dataset
scaler.fit(X_train)

# then we call the transform method to scale both the training and testing data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# a sample output
print(X_train_scaled[0])

model = models.Sequential()

model.add(layers.Dense(8, activation='sigmoid', input_shape=[X_train.shape[1]]))
model.add(layers.Dense(16, activation='sigmoid'))

# output layer
model.add(layers.Dense(1))
keras.optimizers.SGD(lr=0.1)

model.compile(optimizer=SGD(), loss='mse', metrics=['mae'])

tensorborad = TensorBoard(log_dir="logs/{}".format(time()))

history = model.fit(X_train_scaled, y_train, batch_size=64, validation_split=0.2, epochs=50, callbacks=[tensorborad])

print(model.evaluate(X_test_scaled, y_test))