import tensorflow as tf
import numpy as np
import pandas as pd
# token load import
import pickle

# tokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# load model
from keras.models import load_model

# Part model, tokenizer, pickle 불러오기

model = load_model('SaveModel/CommentAi.keras', compile=False)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

with open('SaveModel/CommentAi_Tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

# df = pd.DataFrame({'review': ['완전 최악이네요 울 엄마도 안살듯 ㅋㅋ', '환불해주세요 개극혐', '헐 여기 개갓성비에요']})

# CSV 파일로 저장 (인덱스 없이, 헤더 포함)
# df.to_csv('Data/CommentReview.csv', index=False, encoding='utf-8-sig')

data = pd.read_csv('Data/CommentReview.csv')

# Part preprocessing

# 1. data['review'] column 정규식 적용
data['review'] = data['review'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣0-9 ]', '', regex=True)

# 2. 중복 제거
data.drop_duplicates(subset=['review'], inplace=True)

# 3. 리스트화
datalist = data['review'].tolist()

# 4. tokenizer가지고 정수 스퀸스
seq_text = tokenizer.texts_to_sequences(datalist)

# 5. 길이 제한
pre_text = pad_sequences(seq_text, maxlen=120)

# Part predict

result = model.predict(pre_text)

# Part predict print

# <class 'numpy.ndarray'>
# print(type(result))

print('긍정은 0.5 이상, 부정은 0.5 이하')
        
# f'{i:.2f}' : 소숫점 2자리까지만 출력
for i in result.flatten():
    print(f'{i:.10f}')