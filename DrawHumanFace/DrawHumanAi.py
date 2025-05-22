from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

file_list = os.listdir('img_align_celeba')
# print(file_list)

images = []

for i in file_list[0:50000]:
    # resize 사진 줄이기
    Image_to_number = Image.open('img_align_celeba/' + i).crop((20, 30, 160, 180)).resize((64, 64))
    images.append(np.array(Image_to_number))
    
plt.imshow(images[1])
plt.show()

