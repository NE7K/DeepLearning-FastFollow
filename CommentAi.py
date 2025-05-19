import pandas as pd
import numpy as np
import os
import tensorflow as tf

# 문자 > 정수 import
from keras.preprocessing.text import Tokenizer

# 120자 이내로 데이터 제한할때 나머지 0으로 채울때 사용
# from keras.preprocessing.sequence import pad_sequences

# GPU 컴에서 사용할 경우
from tensorflow.keras.preprocessing.sequence import pad_sequences


# validation말고 test data 떼고 섞어
from sklearn.model_selection import train_test_split

# early stop
from keras.callbacks import EarlyStopping

# label : rating review
raw = pd.read_table('Data/naver_shopping.txt', names=['rating', 'review'])
# print(raw)

# data frame create label, rating이 3보다 크면 1 아니면 0
raw['label'] = np.where( raw['rating'] > 3, 1, 0)

# print(raw)

# 특수문자, 영어 제거 (오타포함) 영어 포함시키리면 A-Za-z
# BUG 스페이스바 오른쪽에 입력하면 스페이스바 살려줌
# BUG regex=True 명시 안해주면 정규식 적용이 안되는거 같음
raw['review'] = raw['review'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣0-9 ]', '', regex=True)

# null none check
# print(raw.isnull().sum())

# 중복값 제거
raw.drop_duplicates(subset=['review'], inplace=True)
# print(raw)

# bag of words
Unique_Text = raw['review'].tolist() # [review context] 로 변환
Unique_Text = ''.join(Unique_Text)
Unique_Text = list(set(Unique_Text))
Unique_Text.sort()
# Note 유니크 문자
# print(Unique_Text[0:100])

# Part 문자 > 정수 tensorflow
# char level True = 글자 단위, False = 단어 단위
# oov_token 나중에 새로운 것이 들어왔을 때 <OOV>로 정의
tokenizer = Tokenizer(char_level=True, oov_token='<OOV>')

# 문자 리스트
context_list = raw['review'].tolist()
tokenizer.fit_on_texts(context_list)

# 유니크 문자 정수 치환되었는지 확인
# print(tokenizer.word_index)
# print(context_list[0:10])

# Note 단어단위로 전처리할 때, 전체 데이터 중 1회이하 출현단어는 index에서 삭제

# Part X 데이터 : 문자 리스트를 알맞은 숫자로 변환
train_seq = tokenizer.texts_to_sequences(context_list)
# print(train_seq)

# Y 데이터
Y = np.array(raw['label'].tolist())
# print(Y[0:10])

# Part X 데이터를 삽입할 때 글자 수를 맞춰주는 것이 중요

# 길이 열 추가
raw['lenght'] = raw['review'].str.len()
# print(raw)

# 최대길이
# print(raw.head())
# 0.500404로 대충 반반 한쪽으로 치우쳐져 있으면, 선플 혹은 악플 중 하나만 구분 가능할지도
# print(raw.describe())

# max         5.000000       1.000000     140.000000 최대 140글자
# 120글자 이내는 몇 개의 데이터가 존재하는지
count = raw['lenght'][raw['lenght'] < 120].count()
# print(count)

# 최대글자 120 나머지 0
X = pad_sequences(train_seq, maxlen=120)

# random_state 42 시드
trainX, valX, trainY, valY = train_test_split(X, Y, test_size=0.2, random_state=42)

# 데이터 쪼갠거 확인
# print(len(trainX))
# print(len(valX))

# Part 모델 생성
model = tf.keras.models.Sequential([
    # 원핫인코딩 대신 쓰는거 (유니크가 3천개잖아;) 글자를 16개의 글자 벡터로 바꿔줌 행렬로
    tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 16),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# model compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# early stop
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

# batch_size basic 32
model.fit(X, Y, validation_data=(valX, valY), batch_size=64, epochs=5, callbacks=[es])

model.save('SaveModel/CommentAi.keras')