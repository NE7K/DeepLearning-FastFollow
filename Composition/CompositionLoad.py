import tensorflow as tf
import numpy as np

pmodel = tf.keras.models.load_model('SaveModel/CompositionLSTM.keras')

text = open('Data/pianoabc.txt', 'r').read()

Unique_text = list(set(text))

Unique_text.sort()

text_to_num = {}
num_to_text = {}

for i, data in enumerate(Unique_text):
    text_to_num[data] = i
    num_to_text[i] = data

number_text = []

for i in text:
    number_text.append( text_to_num[i] )

picklist = number_text[117 : 117+25]
picklist = tf.one_hot(picklist, 31)

#Part shape=(25, 31), shape=(1, 25, 31) : 2차원 데이터를 3차원 데이터로 변경 axis는 
picklist = tf.expand_dims(picklist, axis=0)
# print(picklist)

# Part 
# 0. 첫입력 값 만들기
# 1. predict로 다음문자 예측
# 2. 예측한 다음문자 []저장
# 3. 첫 입력값 앞에 짜르기
# 4. 예측한 다음문자를 뒤에 넣기
# 5. 원핫인코딩하기, expand dims

music = []

for i in range(200):
    predict_value = pmodel.predict(picklist)
    # 예측값 중에서 최대값만 출력
    predict_value = np.argmax(predict_value[0])
    
    # 새로운 예측값
    # new_predict = np.random.choice(Unique_text, 1, p=predict_value[0])
    
    music.append(predict_value)

    # [[]제외하고 나머지 담기]
    next_insert = picklist.numpy()[0][1:]
    # print(next_insert)

    # 예측한 값 원핫인코딩
    one_hot_num = tf.one_hot(predict_value, 31)
    # print('원핫한거', one_hot_num)

    # 맨 앞 값을 뺐으니 예측한 값을 넣어야함
    picklist = np.vstack([ next_insert, one_hot_num.numpy() ])
    picklist = tf.expand_dims(picklist, axis=0)


# print(predict_value)
# print(num_to_text[predict_value])
# print(number_text[117+25])
# print(music)

music_text = []

# 텍스트로 변환
for i in music:
    music_text.append(num_to_text[i])

# 합치기
print(''.join(music_text))