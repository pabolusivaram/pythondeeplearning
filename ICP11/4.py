

import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization
from keras.constraints import maxnorm
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

K.common.image_dim_ordering()
from PIL import Image
from keras.utils import np_utils
import glob
import cv2

train_images = []
import numpy as np

seed = 7
np.random.seed(seed)

for filename in glob.glob('/content/drive/My Drive/Images/Car/*.jpg'):
    img_normal = cv2.imread(filename)
    output = cv2.resize(img_normal, (32, 32))
    train_images.append([output, 0])

for filename in glob.glob('/content/drive/My Drive/Images/Bike/*.jpg'):
    img_normal = cv2.imread(filename)
    output = cv2.resize(img_normal, (32, 32))
    train_images.append([output, 1])

for filename in glob.glob('/content/drive/My Drive/Images/Dog/*.jpg'):
    img_normal = cv2.imread(filename)
    output = cv2.resize(img_normal, (32, 32))
    train_images.append([output, 2])

for filename in glob.glob('/content/drive/My Drive/Images/House/*.jpg'):
    img_normal = cv2.imread(filename)
    output = cv2.resize(img_normal, (32, 32))
    train_images.append([output, 3])
import random

random.shuffle(train_images)

x_train = []
y_train = []
for im, label in train_images:
    x_train.append(im)
    y_train.append(label)

x_train = np.array(x_train).reshape(-1, 32, 32, 3)

type(x_train)
x_train.shape
x_train[0]

test_images = []
for filename in glob.glob('/content/drive/My Drive/Images/Car/*.jpg'):
    img_normal = cv2.imread(filename)
    output = cv2.resize(img_normal, (32, 32))
    test_images.append([output, 0])

for filename in glob.glob('/content/drive/My Drive/Images/Bike/*.jpg'):
    img_normal = cv2.imread(filename)
    output = cv2.resize(img_normal, (32, 32))
    test_images.append([output, 1])

for filename in glob.glob('/content/drive/My Drive/Images/Dog/*.jpg'):
    img_normal = cv2.imread(filename)
    output = cv2.resize(img_normal, (32, 32))
    test_images.append([output, 2])

for filename in glob.glob('/content/drive/My Drive/Images/House/*.jpg'):
    img_normal = cv2.imread(filename)
    output = cv2.resize(img_normal, (32, 32))
    test_images.append([output, 3])

x_test = []
y_test = []
for im, label in test_images:
    x_test.append(im)
    y_test.append(label)

x_test = np.array(x_test).reshape(-1, 32, 32, 3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), input_shape=(x_train.shape[1:]), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

epochs = 10
lrate = 0.001
decay = lrate / epochs
sgd = Adam(lr=lrate)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=128)

import pickle

with open("./Screenshots/shiva_task4_model.pk2", 'wb') as file:
    pickle.dump(model, file)

x = model.predict_classes(x_train[[10], :])

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))



from werkzeug.utils import secure_filename
import joblib
from sss import Flask, request, render_template
import cv2
import numpy as np

# Define a flask app
app = Flask(__name__)

def process_eval(imk):
    output1 = cv2.resize(imk, (32,32))
    output1 = output1.astype('float')
    output1 /= 255.0
    print(type(output1))
    output1 = np.array(output1).reshape(-1, 32, 32, 3)
    classifer = joblib.load("/content/drive/My Drive/Colab Notebooks/shiva_task4_model.pk2")
    x = classifer.predict_classes(output1[[0], :])
    if x[0] == 0:
        result = "The image predicted is a Car"
    elif x[0] ==1:
        result = "The image is predicted a Bike"
    elif x[0] ==2:
        result = "The image is predicted a Dog"
    elif x[0] ==3:
        result = "The image is predicted a House"
    else:
        result = "Image not in trained model"
    return result

@app.route('/', methods=['GET'])
def index():
   return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def handle_form():
    if request.method == 'POST':
        file = request.files['file']
        file.save(secure_filename("save.jpeg"))
        im=cv2.imread("save.jpeg")
        result=process_eval(im)
        return render_template('index.html',result=result)

if __name__ == "__main__":
    app.run()
