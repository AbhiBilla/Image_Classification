#Installing Required Packages
!pip install streamlit --quiet
!pip install pyngrok==4.1.1
from pyngrok import ngrok

#Code for Web app
%%writefile app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

flowers=['daisy','sunflower','rose','dandelion','tulip']
model1=tf.keras.models.load_model('/content/drive/MyDrive/Major Project Billa Abhignan/flowers Model.hdf5')

st.title("Flower Recognizer")
upload = st.sidebar.file_uploader(label='Upload the Image')

if upload is not None:
  img1=np.asarray(bytearray(upload.read()),dtype=np.uint8)
  img1=cv2.imdecode(img1,1)
  img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
  st.image(img1,caption='Uploaded Image',width=300)
  if st.sidebar.button('PREDICT'):
    st.sidebar.write("Result:")
    x = cv2.resize(img1,(48,48))
    x = np.expand_dims(x,axis=0)
    x=x/255
    st.image(x,caption='Processed_Image',width=48)
    y = model1.predict(x)
    label=y
    y=np.argmax(y)
    st.title(flowers[y])
    # print the classification
    for i in range(5):
      out=label[0]
      st.sidebar.title('%s (%.2f%%)' % (flowers[i], out[i]*100))

#Getting URL
!nohup streamlit run app.py &
url = ngrok.connect(port='8501')
url

