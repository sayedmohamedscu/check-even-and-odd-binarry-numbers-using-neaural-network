import keras
import numpy as np
from keras.utils import plot_model
model=0
'''x_train = np.array([[9/15],[8/15],[0/15], [1/15], [11/15], [3/15], [4/15], [5/15], [6/15],[1]])
y_train = np.array([[1,0,0,1],[1,0,0,0],[0,0,0,0], [0,0,0,1], [1,0,1,1], [0,0,1,1], [0,1,0,0],[0,0,1,1], [0,1,1,0],[1,1,1,1]])
x_test = np.array([[10/15], [11/15], [12/15], [13/15], [14/15], [1/15]])
y_test = np.array([[1,0,1,0],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],[0,0,0,1]])'''

y_train = np.array([[0],[0], [1], [1], [1], [0], [1], [0],[1]])
x_train = np.array([[1,0,0,0],[0,0,0,0], [0,0,0,1], [1,0,1,1], [0,0,1,1], [0,1,0,0],[0,0,1,1], [0,1,1,0],[1,1,1,1]])
y_test = np.array([[0],[1], [0], [1], [0], [1]])
x_test = np.array([[1,0,1,0],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],[0,0,0,1]])

print(x_train.shape)
print(y_train[1])
print(x_test.shape)
print(y_test.shape)
x_train=x_train
x_test=x_test
from keras.layers import Dense, Activation
from keras.models import Sequential
model = Sequential()

model.add(Dense(4, input_dim=4,))
#model.add(Dense(4))
model.add(Activation('relu'))
#model.add(Dense(4))
#model.add(Activation('relu'))

#model.add(Dense(4))

#model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))#softmax
#model.add(Activation('softmax'))

model.summary()
model.compile(loss='binary_crossentropy', optimizer='sgd',metrics=
['accuracy'])
history=model.fit(x_train, y_train,
          batch_size=1,
          epochs=10000,
          verbose=1,
          validation_data=(x_test, y_test))

out=np.array([1,0,1,0])
print(out)
out=out.reshape(1,4)
out=model.predict(out)
print (out)

"""*****************************************************"""
"""************RESULT VISUALIZING***********************"""
"""*****************************************************"""

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'c', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'c', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
