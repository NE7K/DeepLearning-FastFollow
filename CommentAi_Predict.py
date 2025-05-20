import tensorflow as tf
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer

# insert data
data = pd.read_table('Data/test.txt', names=['index', 'example_text'])
print(data)

# 정규식 적용, regex=True로 정규식 선언 이거 추가 안하면 띄어쓰기 때문에 오류나는듯
data['example_text'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣0-9 ]', regex=True)

# 중복값 제거, inplace True로 원본 수정
data.drop_duplicates(subset=['example_text'], inplace=True)

# 리스트 변환
new_text = data['example_text'].tolist()

# 시퀸스 길이 제한 - tokenizer


# predict
