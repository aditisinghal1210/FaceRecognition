import numpy as np
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import random

IMG_SIZE=64
TRAIN_TEST_SPLIT=70
NO_OF_EPOCHS=6


data = []
for img in os.listdir("data"):
    path = os.path.join("data", img)
    img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
    name = img.split('.')
    if len(name) < 3:
        continue
    name = name[1] 
    data.append([np.array(img_data), np.array(int(name))])


random.shuffle(data)
Till = 70/100
Till*=len(data)
train = data[: int(Till)]
test = data[int(Till) :]

X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.array([i[1] for i in train])
X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = np.array([i[1] for i in test])

NO_OF_CLASSES = len(np.unique([i[1] for i in data]) )   
print(NO_OF_CLASSES)

model = keras.Sequential([
    keras.layers.Conv2D(16, kernel_size=(3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(NO_OF_CLASSES+1, activation="softmax"),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

y_train = keras.utils.to_categorical(y_train, num_classes=NO_OF_CLASSES+1)
y_test = keras.utils.to_categorical(y_test, num_classes=NO_OF_CLASSES+1)

new_model = model.fit(X_train, y_train, epochs=NO_OF_EPOCHS, validation_data=(X_test, y_test))
model.save('model.h5')
# pred_class=[1,2,3,4]

y_pred_batch = model.predict(X_test)
# predicted_classes = np.argmax(y_pred_batch, axis=1)
# print(predicted_classes)

