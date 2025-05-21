import pandas as pd

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