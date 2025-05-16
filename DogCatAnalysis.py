import tensorflow as tf
import os
import shutil
# print( len ( os.listdir('') ) )

for i in os.listdir('/content/train/'):
    if 'cat' in i:
        shutil.copyfile('/content/train' + i ,'/content/dataset/cat' + i)

# 이미지 숫자화
# tf.keras.preprocessing.image_dataset_from_directory()