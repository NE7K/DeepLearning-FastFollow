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
# print(image_np.shape)

# Part Discriminator : 이미지 진짜인지 가짜인지 구분 모델
discriminator = tf.keras.models.Sequential([
    # 64개의 3x3 필터로 특징 추출, strides로 필터가 2칸씩 이동, padding 출력 크기 유지
    tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=[64, 64, 1]),
    # 음수인 경우 0.2 곱함
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 랜점숫자 100개를 넣으면 이미지 1개
noise_shape = 100

generator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4 * 4 * 256, input_shape=(100,)),
    tf.keras.layers.Reshape((4,4,256)),
    # 이미지 그림 2배로 키웠다가 컨볼루션 적용
    tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    # info Covariate shift 문제를 해결하기 위해서 도입하는 레이어
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.BatchNormalization(),
    # 64, 64, 1 output 내보내기 reshape만쓰면 연관성이 떨어짐
    tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding='same', activation='sigmoid')
])

# print(generator.summary())

GAN = tf.keras.models.Sequential([generator, discriminator])

discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# 학습할 필요가 없음 구분만
discriminator.trainable = False

GAN.compile(optimizer='adam', loss='binary_crossentropy')

def predict_print():
    # 랜덤숫자 (-1, 1)까지 균일하게 랜덤으로 100개를 8세트 뽑기
    random_number = np.random.uniform(-1, 1, size=(10,100))

    predict_data = generator.predict(random_number)
    # print(predict_data.shape)

    for i in range(10):
        # 이미지 저장으로 딥러닝 종료 방지
        img = predict_data[i].reshape(64, 64)
        plt.imsave(f'trainingimg/{i}.jpg', img, cmap='gray')
        # 2x5 사이즈로 배치
        # plt.subplot(2,5, i+1)
        # 이미지 보여주기
        # plt.imshow(predict_data[i].reshape(64, 64), cmap='gray')
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.show()

x_data = images

# discriminator training
# discriminator.train_on_batch(진짜 사진 128장, 1로 마킹한 정답)
# discriminator.train_on_batch(가짜 사진 128장, 0으로 마킹한 정답)

for j in range(300):
    
    # print
    print(f'현재 epoch : {j}')
    
    predict_print()
    
    for i in range(50000//128):
        
        if i % 100 == 0:
            print(f'현재 몇 번째 batch : {i}')
        
        real_pic = x_data[i * 128 : (i+1)*128]
        one_data = np.ones(shape=(128, 1))

        loss1 = discriminator.train_on_batch(real_pic, one_data)

        random_number = np.random.uniform(-1, 1, size=(128,100))
        fake_pic = generator.predict(random_number)
        zero_data = np.zeros(shape=(128, 1))

        loss2 = discriminator.train_on_batch(fake_pic, zero_data)

        random_number = np.random.uniform(-1, 1, size=(128,100))
        one_data = np.ones(shape=(128, 1))

        loss3 = GAN.train_on_batch(random_number, one_data)
        
    print(f'이번 epochs 최종 loss는 Discriminator : {loss1+loss2}, GAN : {loss3}')