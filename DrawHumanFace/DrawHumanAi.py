from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# image file list로 불러옴
file_list = os.listdir('img_align_celeba')
# print(file_list)

# image 담을 리스트
images = []

# file list 0부터 5만개까지만 사용할거임
for i in file_list[0:50000]:
    # 이미지 숫자화 시킴 resize 사진 줄이기, crop 자르고, convert L은 흑백 변환
    Image_to_number = Image.open('img_align_celeba/' + i).crop((20, 30, 160, 180)).convert('L').resize((64, 64))
    # append하기 전에 np array에 집어넣음
    images.append(np.array(Image_to_number))
    
# plt.imshow(images[1])
# plt.show()

# 이미지 255로 나눔
images = np.divide(images, 255)
# 이미지를 레이어에 넣기 위해서는 4차원이 필요한데, 흑백은 3차원이기 때문에 뒤에 1 붙여서 흑백 4차원 행렬 만들어주기
images = images.reshape(50000, 64, 64, 1)

# 이미지 넘파이에 저장해서 쉐입 출력해보기
image_np = np.array(images)
print(image_np.shape)

# Part Discriminator

discriminator = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=[64,64,1]),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Conv2D(64, (3,3), strides=(2, 2), padding='same'),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1, activation='sigmoid')
]) 

# 
noise_shape = 100

generator = tf.keras.models.Sequential([
  tf.keras.layers.Dense(4 * 4 * 256, input_shape=(100,) ), 
  tf.keras.layers.Reshape((4, 4, 256)),
  tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same'),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same'),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same'),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding='same', activation='sigmoid')
])

print(generator.summary())

GAN = tf.keras.models.Sequential([generator, discriminator])

discriminator.compile(optimizer='adam', loss='binary_crossentropy')

discriminator.trainable = False

GAN.compile(optimizer='adam', loss='binary_crossentropy')

