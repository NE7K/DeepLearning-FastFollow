import pandas as pd
import tensorflow as tf
import numpy as np

data = pd.read_csv('Data/train.csv')

# .isnull() : null이면 True, .sum() 개수
print(data.isnull().sum())

# 빈칸에서 나이는 평균값
print(data['Age'].mean())

# 평균값 반올림해서 빈칸에 채우기
data['Age'] = data['Age'].fillna(value=30)

# 탑승구 최빈값
print(data['Embarked'].mode())

data['Embarked'] = data['Embarked'].fillna(value='S')

print(data.isnull().sum())

y = data.pop('Survived')

# normalization
Fare_Preprocessing = tf.keras.layers.Normalization(axis=None)
Fare_Preprocessing.adapt(np.array(data['Fare']))

Sib_Preprocessing = tf.keras.layers.Normalization(axis=None)
Sib_Preprocessing.adapt(np.array(data['SibSp']))

Parch_Preprocessing = tf.keras.layers.Normalization(axis=None)
Parch_Preprocessing.adapt(np.array(data['Parch']))

Pclass_Preprocessing = tf.keras.layers.Normalization(axis=None)
Pclass_Preprocessing.adapt(np.array(data['Pclass']))

# print(Pclass_Preprocessing(np.array(data['Pclass'])))

# 비슷한 숫자 묶음
Age_Preprocessing = tf.keras.layers.Discretization(bin_boundaries=[10, 20, 30, 40, 50, 60])

# Age 변환 체크
# print(Age_Preprocessing(np.array(data['Age'])))

# categoryencoding은 원핫인코딩해줌 카테고리별로

# 문자 원핫인코딩
Sex_Preprocessing = tf.keras.layers.StringLookup(output_mode='one_hot')
Sex_Preprocessing.adapt(np.array(data['Sex']))

Embarked_Preprocessing = tf.keras.layers.StringLookup(output_mode='one_hot')
Embarked_Preprocessing.adapt(np.array(data['Embarked']))

# Create Embedding layers : 종류가 많아
# 데이터를 정수로 치환해줌
Ticket_Preprocessing = tf.keras.layers.StringLookup()
Ticket_Preprocessing.adapt(np.array(data['Ticket']))

# 유니크한 데이터 수
print(len(data['Ticket'].unique()))

Ticket_Preprocessing = tf.keras.layers.Embedding(len(data['Ticket'].unique()) + 1, 10)

# functional api 사용할거라서 input 생성
input_fare = tf.keras.Input(shape=(1,), name='Fare')
input_parch = tf.keras.Input(shape=(1,), name='Parch')
input_sibsp = tf.keras.Input(shape=(1,), name='SibSp')
input_pclass = tf.keras.Input(shape=(1,), name='Pclass')
input_age = tf.keras.Input(shape=(1,), name='Age')
# 문자는 , dtype=tf.string 붙임
input_sex = tf.keras.Input(shape=(1,), name='Sex', dtype=tf.string)
input_embarked = tf.keras.Input(shape=(1,), name='Embarked', dtype=tf.string)
input_ticket = tf.keras.Input(shape=(1,), name='Ticket', dtype=tf.string)

