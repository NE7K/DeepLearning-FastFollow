import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

# Load the Fashion MNIST dataset
# Dtype은 튜플
( trainX, trainY ), ( testX, testY ) = tf.keras.datasets.fashion_mnist.load_data()

# info image data 전처리 0~255를 넣는게 아니라 0~1로 압축해서 넣음
trainX = trainX / 255.0
testX = testX / 255.0

# 자료 타입 맞추기
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

model = tf.keras.Sequential([
    # 다른 패널 사용, 32개 복사본, (3,3) kernal 사이즈, 이미지는 음수가 없기 때문에 relu, shape 흑백은 1 color은 3
    tf.keras.layers.Conv2D( 32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    # (2,2) = pooling size
    tf.keras.layers.MaxPooling2D( (2,2) ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile( loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# save log
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='LogFile/Log{}'.format('_Model_' + str(int(time.time()))) )

model.fit(trainX, trainY, validation_data=(testX, testY), epochs=3, callbacks=[tensorboard])

# Part hidden layers fix
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D( 32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D( (2,2) ),
    tf.keras.layers.Conv2D( 64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D( (2,2) ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile( loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# save log
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='LogFile/Log{}'.format('_Model_' + str(int(time.time()))) )

# Part callback patience early stop, 3번 loss 감소가 진전이 없으면 실행 accuracy는 mode max
es = tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=3, mode='min' )

model.fit(trainX, trainY, validation_data=(testX, testY), epochs=3, callbacks=[tensorboard, es])

# Part tensorboard --logdir LogFile : log board print