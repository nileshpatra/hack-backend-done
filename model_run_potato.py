import os
os.environ['KERAS_BACKEND'] = 'theano'
import keras
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import Sequential
from keras.layers import Dense , Flatten , Convolution2D , MaxPooling2D , Conv2D , Dropout 
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator



from keras.preprocessing.image import ImageDataGenerator


import pickle
filename = 'ker_model_potato.sav'
loaded_model = pickle.load(open(filename ,'rb'))

fl = open('pred.html' , 'r')
path = fl.read()
fl.close()
print(path)

train_datagen = ImageDataGenerator(
rescale = 1./255,
shear_range = 0.2,
zoom_range=0.2 ,
horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


xtest = train_datagen.flow_from_directory(path
                                          , target_size = (64,64),
                                          batch_size = 32 ,
                                          class_mode = 'binary'
                                        )
_ , (image , label) = next(enumerate(xtest))

result = loaded_model.predict(image)
print(result)
var = ''
if result.argmax() == 0:
    var = 'healthy'
elif result.argmax() == 1:
    var = 'rot'
elif result.argmax() == 2:
    var = 'rust'
else:
    var = 'scab'

print(var)
with open('P4Output.html', 'w') as f:
    f.write(s)
# file = open('pred.txt' , 'r')
# print(file.read())

