import streamlit as st
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("./trained_model.keras")

#Tensorflow Model Prediction
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(224,224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","plant classification"])

#Main Page
if(app_mode=="Home"):
    st.header("ornamintal plant classification system")
    image_path = "home_page.jpeg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the ornamintal plant classification System! 🌿🔍
    
    Our mission is to help in identifying ornamintal plants efficiently. Upload an image of a plant, and our system will analyze it to know what tyoe of plant it is.

    ### How It Works
    1. **Upload Image:** Go to the **plant classification** page and upload an image of a plant.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify the plant.
    3. **Results:** View the results .

    ### Get Started
    Click on the **plant classification** page in the sidebar to upload an image.

    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 3K rgb images of different ornamintal plants and categorized into 5 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 5 test images is created later for prediction purpose.
                #### Content
                1. train (2398 images)
                2. test (5 images)
                3. validation (469 images)

                """)

#Prediction Page
elif(app_mode=="plant classification"):
    st.header("plant classification")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image, use_container_width=True)
    #Predict button
    if(st.button("Predict")):
        result_index = model_prediction(test_image)
        st.write("Our Prediction")
        #Reading Labels
        class_name = ['Damask Rose', 'Echevria Flower', 'Mirabilis Jalapa', 'Rain Lily','Zinnia Elegans']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))