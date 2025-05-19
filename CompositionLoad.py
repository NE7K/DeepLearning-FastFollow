import tensorflow as tf

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

print(number_text[117 : 117+25])