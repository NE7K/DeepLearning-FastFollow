text = open('Data/pianoabc.txt', 'r').read()

# print(text)

# bag of words, set : 중복없는 list
Unique_text = list(set(text))

# 정렬
Unique_text.sort()

# Part utilities
text_to_num = {}
num_to_text = {}

# alphabet > number change
for i, data in enumerate(Unique_text):
    text_to_num[data] = i
    num_to_text[i] = data
    
# print(text_to_num)
print(text_to_num['3'])

# number list
number_list = []

# text > number change
for i in text:
    number_list.append( text_to_num[i] )

# print(number_list)