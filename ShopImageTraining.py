import tensorflow as tf
import matplotlib.pyplot as plt

# Load the Fashion MNIST dataset
# Dtype은 튜플
( trainX, trainY ), ( testX, testY ) = tf.keras.datasets.fashion_mnist.load_data()

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
    # Note create hidden layers 
    # info summary 사용 시 input_shape=(28,28) 필요
    tf.keras.layers.Dense(128, input_shape=(28,28), activation='relu'),
    # relu는 음수는 다 0으로 만들어주고 convolution layer에서 자주 사용
    tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Flatten(),
    # last layers 노드수는 카테고리 갯수만큼
    # 확률 0 ~ 1 softmax | sigmoid는 binary 예측 문제에 사용 0or1
    tf.keras.layers.Dense(10, activation='softmax')
])

# model 요약본
model.summary()

# info 원핫인코딩에는 categorical crossentropy loss, tf.keras.utils.to_categroical()사용
# categorie 예측일 때에는 아래 loss 사용
model.compile( loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=5)