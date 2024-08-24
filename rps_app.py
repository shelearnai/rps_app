from tensorflow import keras
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
#from keras.preprocessing import image
import keras.utils as image
import numpy as np

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array

st.set_page_config(page_title='Rock Paper Scissor Classification')
st.title('Rock Paper Scissor Classification')
classlabel=['rock','paper','scissors']

@st.cache(allow_output_mutation=True)
def get_best_model():
    model = keras.models.load_model('rps_model.h5',compile=False)
    model.make_predict_function()          # Necessary
    print('Model loaded. Start serving...')
    return model

st.subheader('Classify the image')
image_file = st.file_uploader('Choose the Image', ['jpg', 'png'])
if image_file:
    print(image_file)
    st.image(image_file, caption='RPS Image', use_column_width=True)
    img=load_img(image_file,target_size=(150,150))
    img_arr=img_to_array(img)
    img_arr=np.expand_dims(img_arr,axis=0)  #AI -> Tabnine
    model=get_best_model()
    classes=model.predict(img_arr)
    label_position=np.argmax(classes)  
    pred_value=classlabel[label_position]
    st.markdown(f'<h3>The image is predicted as {pred_value}.</h3>', unsafe_allow_html=True)
    



