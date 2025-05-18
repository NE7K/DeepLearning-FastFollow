import numpy as np
from sklearn import metrics
import tensorflow as tf
import os
import shutil
import matplotlib.pyplot as plt

# tensorboard 사용 시 필요한 import
import time

# Part data set 150, 150으로 생성
# os.mkdir('dataset2')
# os.mkdir('dataset2/cat')
# os.mkdir('dataset2/dog')

# for i in os.listdir('train/'):
#     if 'cat' in i:
#         shutil.copyfile('train/' + i ,'dataset2/cat/' + i)
#     if 'dog' in i:
#         shutil.copyfile('train/' + i, 'dataset2/dog/' + i)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/',
    image_size=(150, 150),
    batch_size=64,
    subset='training',
    validation_split=0.2,
    seed=1234
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/',
    image_size=(150, 150),
    batch_size=64,
    subset='validation',
    validation_split=0.2,
    seed=1234
)

print(train_ds)

def preprocessing(i, result):
    i = tf.cast( i/255.0, tf.float32 )
    return i, result

train_ds = train_ds.map(preprocessing)
val_ds = val_ds.map(preprocessing)

# Part
# info from import 제외함
# input shape 강제로 맞추고 include top false로 출력층 제외, weights 값 파일로 적용할 예정이니 제외
inception_model = tf.keras.applications.inception_v3.InceptionV3( input_shape=(150,150,3), include_top=False, weights=None)

# load model
inception_model.load_weights('inception_v3.h5')

# inception_model.summary()

# Part last Dense layer외 학습 금지 설정
for i in inception_model.layers:
    # print(i)
    # w 값 고정
    i.trainable = False

# Part fine tuning : mixed6 이후부터는 weight 값 변경되게
unfreeze = False
for i in inception_model.layers:
    if i.name == 'mixed6':
        unfreeze = True
    if unfreeze == True:
        i.trainable = True
    
# Part model 튜닝 - 원하는 곳에서 자르기
last_layer = inception_model.get_layer('mixed7')

print(last_layer)
print(last_layer.output)
# BUG output_shape가 아니라 output.shape임
print(last_layer.output.shape)

# Part last layer랑 이어주기 funcational api
x1 = tf.keras.layers.Flatten()(last_layer.output)
x2 = tf.keras.layers.Dense(1024, activation='relu')(x1)
x3 = tf.keras.layers.Dropout(0.2)(x2)
x4 = tf.keras.layers.Dense(1, activation='sigmoid')(x3)

# Part
# model 생성, 1 layer : inception model
model = tf.keras.Model(inception_model.input, x4)

# Part fine tuning, lr 0.00001로하면 w값이 조금씩 바뀌겠지
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.00001), metrics=['acc'])

model.fit(train_ds, validation_data=val_ds, epochs=2)