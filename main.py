from helper import *

import streamlit as st

import os
 

from PIL import Image

st.title('Pet Popularity Predictor')

def save_uploaded_file(uploaded_file):

    try:

        with open(os.path.join('images',uploaded_file.name),'wb') as f:

            f.write(uploaded_file.getbuffer())

        return 1    

    except:

        return 0
     

uploaded_file = st.file_uploader("Upload Image")

# text over upload button "Upload Image"

if uploaded_file is not None:

    if save_uploaded_file(uploaded_file): 

        # display the image

        display_image = Image.open(uploaded_file)

        st.image(display_image)

        prediction = predictor(os.path.join('images',uploaded_file.name))

        os.remove('images/'+uploaded_file.name)

        # deleting uploaded saved picture after prediction

        st.text('Predictions :-')
        st.success('The output is {}'.format(prediction))
