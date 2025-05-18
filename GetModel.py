import tensorflow as tf

GetModel = GetModel = tf.keras.models.load_model('SaveModel/DogCatAnalysis1')

GetModel.summary()
# GetModel.evaluate()