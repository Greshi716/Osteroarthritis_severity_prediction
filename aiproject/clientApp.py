import streamlit as st
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model
model1=load_model('model1.h5')
CLASS_NAMES=['level 0', 'level 1','level 2','level 3','level 4']
st.title("Osteroarthristis level prediction")
st.markdown("Upload your image")
vehicle_image = st.file_uploader('Upload Image')
submit=st.button('Predict')

if submit:
    if vehicle_image is not None:
        file_bytes=np.asarray(bytearray(vehicle_image.read()), dtype = np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image)
        opencv_image = opencv_image.astype('float32') / 255.0
        opencv_image.shape=(1,224,224,3)
        model_rf = pickle.load(open('trained_final.sav', 'rb'))
        model = load_model('modelf.h5')
        print(opencv_image)
        features = model.predict(opencv_image)
        features = features.reshape((features.shape[0], -1))
        prediction = model_rf.predict(features)
        print(prediction)
        Y_pred = model1.predict(opencv_image)
        print(Y_pred)
        st.success(str("The image is "+   CLASS_NAMES[np.argmax(Y_pred)]))
