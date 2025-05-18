import numpy as np
from sklearn import metrics
import tensorflow as tf
import os
import shutil
import matplotlib.pyplot as plt

# tensorboard 사용 시 필요한 import
import time

# print( len ( os.listdir('') ) )

# os.mkdir('dataset')
# os.mkdir('dataset/cat')
# os.mkdir('dataset/dog')

# exit()

# for i in os.listdir('train/'):
#     if 'cat' in i:
#         shutil.copyfile('train/' + i ,'dataset/cat/' + i)
#     if 'dog' in i:
#         shutil.copyfile('train/' + i, 'dataset/dog/' + i)
        
# exit()

# 이미지 숫자화 80%
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/',
    # Data preprocessing
    image_size=(64, 64),
    # image 2만장 한 번에 넣지 않고 64개씩
    batch_size=64,
    # 데이터를 0.2개로 쪼개서 validation으로
    subset='training',
    validation_split=0.2,
    seed=1234
)

# 20%
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/',
    # Data preprocessing
    image_size=(64, 64),
    # image 2만장 한 번에 넣지 않고 64개씩
    batch_size=64,
    # 데이터를 0.2개로 쪼개서 validation으로
    subset='validation',
    validation_split=0.2,
    seed=1234
)

print(train_ds)

# Part image 0~1로 압축
def preprocessing(i, result):
    i = tf.cast( i/255.0, tf.float32 )
    return i, result

train_ds = train_ds.map(preprocessing)
val_ds = val_ds.map(preprocessing)

# i 64개 데이터 result 64개 y data
for i, result in train_ds.take(1):
    print(i)
    print(result)
#     plt.imshow( i[0].numpy().astype('uint8') )
#     plt.show()

model = tf.keras.Sequential([
    
    # model에 image 넣기 전에 이미지 증강
    # 가로로 뒤집기, input shape은 첫 번째 레이어에
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=(64, 64, 3)),
    # 뒤집기
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    # 줌
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),

    tf.keras.layers.Conv2D( 32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D( (2, 2) ),
    tf.keras.layers.Conv2D( 64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D( (2, 2) ),
    # info overfitting - node 20% drop out
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D( 128, (3, 3), padding='same', activation='relu',),
    tf.keras.layers.MaxPooling2D( (2, 2) ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# info Create log file
tensorboard = tf.keras.callbacks.TensorBoard( log_dir='LogFile/Log{}'.format('_Model_' + str(int(time.time()))) )

model.fit(train_ds, validation_data=val_ds, epochs=1, callbacks=[tensorboard])

# save model
# model.save('SaveModel/DogCatAnalysis1.keras')

# get model 
# GetModel = tf.keras.models.load_model('SaveModel/DogCatAnalysis1.keras')

# GetModel.summary()
# GetModel.evaluate()