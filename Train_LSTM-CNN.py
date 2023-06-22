import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.layers import LSTM, Conv2D, MaxPooling2D
from keras.layers import *
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split

# Read csv file
forward_df = pd.read_csv("datacar/Dstart.txt")
backward_df = pd.read_csv("datacar/Dback1.txt")
left_df = pd.read_csv("datacar/Dleft.txt")
right_df = pd.read_csv("datacar/Dright.txt")
backward1_df = pd.read_csv("datacar/Dstop.txt")
not_df = pd.read_csv("Dnot.txt")

forward2_df = pd.read_csv("datacar/Ustart.txt")
backward2_df = pd.read_csv("datacar/Uback.txt")
left2_df = pd.read_csv("datacar/Uleft.txt")
right2_df = pd.read_csv("datacar/Uright.txt")
backward3_df = pd.read_csv("datacar/Ustop.txt")
not2_df = pd.read_csv("UNot.txt")

not1_df = pd.read_csv("Not.txt")

X = []
y = []
no_of_timesteps = 20

# Bỏ cột đầu tiên vi chỉ là số index
# FORWARD
dataset = forward_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([1,0,0,0,0])

dataset = forward2_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([1,0,0,0,0])
# BACKWARD
dataset = backward1_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,1,0,0,0])

dataset = backward_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,1,0,0,0])

dataset = backward2_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,1,0,0,0])

dataset = backward3_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,1,0,0,0])
# LEFT
dataset = left_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,0,1,0,0])

dataset = left2_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,0,1,0,0])
# RIGHT
dataset = right_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,0,0,1,0])

dataset = right2_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,0,0,1,0])
# NOTHING
dataset = not_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,0,0,0,1])

dataset = not1_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,0,0,0,1])

dataset = not2_df.iloc[:,1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i-no_of_timesteps:i,:])
    y.append([0,0,0,0,1])

# Convert to numpy array
X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
# Input and Output Dimensions
time_period, sensors = X_train.shape[1], X_train.shape[2]
# reshaping data
input_shape = time_period * sensors
print("Input Shape: ", input_shape)
X_train = X_train.reshape(X_train.shape[0], input_shape)
X_test = X_test.reshape(X_test.shape[0], input_shape)
print(X_train.shape)
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')
X_test = X_test.astype('float32')
y_test = y_test.astype('float32')
print(X_train.shape)
# Create model
model = Sequential()
model.add(tf.keras.Input(shape=(input_shape, 1)))
model.add(tf.keras.layers.BatchNormalization())
model.add(Bidirectional(LSTM(32,return_sequences = True,
                             kernel_initializer= tf.initializers.GlorotUniform(seed = 1))))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32,return_sequences= True,
                             kernel_initializer= tf.initializers.GlorotUniform(seed = 1))))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32,return_sequences= True,
                             kernel_initializer= tf.initializers.GlorotUniform(seed = 1))))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32,return_sequences= True,
                             kernel_initializer= tf.initializers.GlorotUniform(seed = 1))))
model.add(Dropout(0.2))

# Add CNN layers

model.add(Conv1D(filters=64,kernel_size=3, activation='relu', strides=2))

model.add(MaxPool1D(pool_size=4, padding='same'))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu', strides=2))

model.add(GlobalAveragePooling1D())

model.add(Dense(units=64, activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(units=128, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(units=64, activation = "relu"))
model.add(Dropout(0.5))

model.add(Dense(units = 5, activation="softmax"))

model.compile(optimizer="adam", metrics = ['accuracy'], loss = "categorical_crossentropy")

print(len(model.layers))

lrd = ReduceLROnPlateau(monitor = 'val_loss',patience = 8,verbose = 1,factor = 0.50, min_lr = 0.00001)

check_point = ModelCheckpoint('CAR-CP.h5', save_best_only= True, mode = 'auto')


# Save model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[lrd, check_point])
model.save("CAR-CTRL-20ts.h5")

# PLot
# summarize history for accuracy
plot1 = plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
#plt.show()
# summarize history for loss
plot2 = plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()