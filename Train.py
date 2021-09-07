#importing Libraries Required
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input,decode_predictions
from sklearn.metrics import accuracy_score
from keras.preprocessing import image

#Loading Data
train=pd.read_csv('/content/drive/MyDrive/Major Project Billa Abhignan/train.csv')
test=pd.read_csv('/content/drive/MyDrive/Major Project Billa Abhignan/test.csv')

#Preparing Data
x_train=train.iloc[:,1:]
x_test=test.iloc[:,1:]
y_train=train['label']
y_test=test['label']

y_train=y_train.astype('int64')
y_test=y_test.astype('int64')

flowers=['daisy','sunflower','rose','dandelion','tulip']

#Visualizing Data
x_train=np.array(x_train).reshape(-1,48,48,3)
x_test=np.array(x_test).reshape(-1,48,48,3)

#Data Augmentation
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

#Loading the model
model=keras.applications.DenseNet121(input_shape=(48,48,3),
                  weights='imagenet', # loading pre-trained weights
                  include_top=False) # do not include image classification layer at the top

#Fine-Tuning the model
model.trainable=False                 # freezing the base model weights

#our data is normalised so no normalization layer is required

#our model consists of batch normalization layer ,so we need to run them in inference mode(i.e trainable=False)
#so that they do not cause any trouble
inputs=keras.Input(shape=(48,48,3))# creating a input layer
x=data_augmentation(inputs)

x=model(x,training=False) # 'x' consists output from data augmentation layer

x=layers.GlobalAveragePooling2D()(x) # adding a pooling layer on top of our model 

x=layers.Dropout(0.2)(x)

outputs=layers.Dense(5,activation='softmax')(x)

model=keras.Model(inputs,outputs)

model.summary()

#Model Training
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#Model Fit
model.fit(x_train,y_train,epochs=10,validation_split=0.2)

#Fine-Tuning with smaller Learning Rate
model.trainable=True

model.compile(optimizer=keras.optimizers.Adam(1e-5),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#Model Fit
model.fit(x_train,y_train,epochs=10,validation_split=0.2)

#Saving Model
model.save('flowers Model.hdf5')