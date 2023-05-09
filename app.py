
'''
Project - MLOps_POC
User Interace using Streamlit
'''

import streamlit as st
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the Keras model
model = tf.keras.models.load_model('model.h5')

def predict_class(image_path):
    # Load and preprocess the image
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Make predictions using the model
    predictions = model.predict(image)

    # Convert the predictions to class names
    class_names = ['cat', 'dog'] # Add your own class names here
    predicted_class = class_names[np.argmax(predictions)]

    return predicted_class

def main():
    st.title("Image Classifier")

    # Upload an image file
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    # Check if the file was uploaded
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Create a button to classify the uploaded image
        if st.button('Classify Image'):
            # Get the predicted class for the image
            predicted_class = predict_class(uploaded_file)

            # Display the result prompt
            st.success(f'The image is a {predicted_class}!')

if __name__ == "__main__":
    main()
