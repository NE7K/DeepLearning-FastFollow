import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load the Fashion MNIST dataset
# Dtype은 튜플
( trainX, trainY ), ( testX, testY ) = tf.keras.datasets.fashion_mnist.load_data()

# info image data 전처리 0~255를 넣는게 아니라 0~1로 압축해서 넣음
trainX = trainX / 255.0
testX = testX / 255.0

# 자료 타입 맞추기
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

# print(trainX[0])

# # (60000, 28, 28) 뒤부터 해석
# print(trainX.shape)

# plt.imshow(trainX[0])
# # 흑백
# plt.gray()
# # color 수치화
# plt.colorbar()
# plt.show()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot']

# Part create model -> compile -> fit
model = tf.keras.Sequential([
    # 다른 패널 사용, 32개 복사본, (3,3) kernal 사이즈, 이미지는 음수가 없기 때문에 relu, shape 흑백은 1 color은 3
    tf.keras.layers.Conv2D( 32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    # (2,2) = pooling size
    tf.keras.layers.MaxPooling2D( (2,2) ),
    tf.keras.layers.Flatten(),
    # Note create hidden layers 
    # info summary 사용 시 input_shape=(28,28) 필요
    # tf.keras.layers.Dense(128, input_shape=(28,28), activation='relu'),
    # relu는 음수는 다 0으로 만들어주고 convolution layer에서 자주 사용
    tf.keras.layers.Dense(64, activation='relu'),
    # last layers 노드수는 카테고리 갯수만큼
    # 확률 0 ~ 1 softmax | sigmoid는 binary 예측 문제에 사용 0or1
    tf.keras.layers.Dense(10, activation='softmax')
])

# call back function
SaveModelWeight = tf.keras.callbacks.ModelCheckpoint(
    # 덮어쓰기가 싫으면 mnist{epoch} 사용
    filepath='Checkpoint/mnist',
    # validation 값이 최대치만 저장
    monitor='val_acc',
    mode='max',
    # weight only save
    save_weights_only=True,
    # epoch 끝날 때마다 저장
    save_freq='epoch'
)

# model 요약본
model.summary()

# info 원핫인코딩에는 categorical crossentropy loss, tf.keras.utils.to_categroical()사용
# categorie 예측일 때에는 아래 loss 사용
model.compile( loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=3, callbacks=[SaveModelWeight])

# model 평가
# score = model.evaluate( testX , testY)
# print(score)

# model.save('SaveModel/ShopImageTraining')

# GetModel = tf.keras.models.load_model('SaveModel/ShopImageTraining')
# GetModel.summary()
# GetModel.evaluate(testX, testY)