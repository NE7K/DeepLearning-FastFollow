import pandas as pd
import tensorflow as tf
import numpy as np
from keras.models import load_model

# Part 데이터

raw = pd.read_csv('Data/test.csv')
print(raw)

model = load_model('SaveModel/DeadProbability.keras')

# Part 전처리


# Part predict

