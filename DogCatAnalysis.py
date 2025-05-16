from numpy import take
from sklearn import metrics
import tensorflow as tf
import os
import shutil
import matplotlib.pyplot as plt

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
    tf.keras.layers.Conv2D( 32, (3, 3), padding='same', activation='relu', input_shape=(64, 64, 3)),
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
model.fit(train_ds, validation_data=val_ds, epochs=5)