# import necessary libraries such as keras and tensorflow
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

# import sklearm and matplotlib
import numpy as np # linear algebra
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical

# initialize the batch size
batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print the shape of the dataset
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices i.e. convert array into one-hot vector encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# MLP architecture
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# fit the model
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

# evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)

# print the accuracy and loss
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# plot the loss and sccuracy
y_hat = model.predict(x_test)
y_pred = np.argmax(y_hat, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)
print(cm)

plt.plot(history.history['loss'], color='b',label='Training loss')
plt.plot(history.history['val_loss'], color='r',label='Validation loss')
plt.show()
plt.plot(history.history['acc'], color='b',label='Training accuracy')
plt.plot(history.history['val_acc'], color='r',label='Validation Accuracy')
plt.show()

model.summary()
