import pandas as pd
import numpy as np
import os
import tensorflow as tf

# label 부여
raw = pd.read_table('Data/naver_shopping.txt', names=['rating', 'review'])
print(raw)