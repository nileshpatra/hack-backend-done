import os
os.environ['KERAS_BACKEND'] = 'theano'
import keras
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')
import pickle
filename = 'untr.sav'
loaded_model = pickle.load(open(filename ,'rb'))

pth = open('pred.txt' , 'r')
path = pth.read()
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
rescale = 1./255,
shear_range = 0.2,
zoom_range=0.2 ,
horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

xtest = test_datagen.flow_from_directory(path
                                          , target_size = (64,64),
                                          batch_size = 32 ,
                                          class_mode = 'binary'
                                        )
_ , (images  , labels) = next(enumerate(xtest))
predict = loaded_model.predict(images)

file = open('finaltxt.txt','w+')
file.write(str(predict.argmax()))
file = open('finaltxt.txt' , 'r')
print('YOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO' , file.read())