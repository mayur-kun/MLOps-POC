'''
Project - MLOps_POC
User Interace using Streamlit
'''

import streamlit as st

def main():
    st.title("Upload Image")

    # Upload an image file
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    # Check if the file was uploaded
    if uploaded_file is not None:
        # Read the contents of the file
        image = uploaded_file.read()

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Create a button to process the uploaded image
        if st.button('Process Image'):
            # Process the image here
            st.write('Image processed successfully!')

if __name__ == "__main__":
    main()
