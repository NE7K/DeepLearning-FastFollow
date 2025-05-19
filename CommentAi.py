import pandas as pd
import numpy as np
import os
import tensorflow as tf

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
tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, oov_token='<OOV>')

# 문자 리스트
context_list = raw['review'].tolist()
tokenizer.fit_on_texts(context_list)

print(tokenizer.word_index)

# Note 단어단위로 전처리할 때, 전체 데이터 중 1회이하 출현단어는 index에서 삭제