import numpy as np
import tensorflow as tf

text = open('Data/pianoabc.txt', 'r').read()

# print(text)

# bag of words, set : 중복없는 list
Unique_text = list(set(text))

# 정렬
Unique_text.sort()

# Part utilities
text_to_num = {}
num_to_text = {}

# alphabet > number change
for i, data in enumerate(Unique_text):
    text_to_num[data] = i
    num_to_text[i] = data
    
# print(text_to_num)
# print(text_to_num['3'])

# number list
number_text = []

# text > number change
for i in text:
    number_text.append( text_to_num[i] )

# print(number_list)

# Part Data set
X = []
Y = []

# len(number_text) 숫자화 데이터의 길이만큼 반복문을 돌리는데, 마지막 +25는 없는 수니 - 25한 횟수만큼 반복문
for i in range(0, len(number_text) - 25):
    X.append(number_text[i : i+25])
    Y.append(number_text[i + 25])
    
print(X[0 : 5])
print(Y[0 : 5])

print( np.array(X).shape )

# 0을 31개로하는 이유는 유니크한 문자가 31개
print(len(Unique_text))

# Part 원핫인코딩
X = tf.one_hot(X, 31)
Y = tf.one_hot(Y, 31)

# Part create model
model = tf.keras.models.Sequential([
    # 유사품 GRU, input shape 하나의 데이터의 shape
    # LSTM 중첩하고 싶으면 return_sequnences 파라미터 필요
    tf.keras.layers.LSTM(100, input_shape=(25, 31)),
    # softmax인 이유는 카테고리 크로스인트로피는 softmax
    tf.keras.layers.Dense(31, activation='softmax')
])

# 원핫인코딩이 안되어 있으면 sparse 사용해야 함
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# batch size 64개 데이터를 넣고 w값 업데이트, verbose는 학습중 출력되는거 줄이는거, verbose=2
model.fit(X, Y, batch_size=64, epochs=50)

model.save('SaveModel/CompositionLSTM.keras', save_format='keras')