import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model_6k.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #this for convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

#sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",("Home","About","Disease Recognition"))

#Home Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "Home Paddy.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    ## Welcome to the Plant Disease Recognition System (DR AZLIZA & DR FAIZAL) 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Page
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
    ### About Dataset
    This study investigates the application of deep learning, specifically convolutional neural networks (CNNs), to automate the classification of paddy leaf diseases, 
    addressing a significant challenge in agriculture. Six common diseases (bacterial leaf blight, bacterial leaf streak, brown spot, dead heart, hispa, and leaf blast) 
    were targeted using a curated dataset enriched through augmentation techniques and analysed with exploratory data analysis (EDA).
    ### Content
    1. Train (4800 Images)
    2. Validation (1200 Images)        
""")
    
#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose Your Image:")
    if(st.button("Show Image")):
        st.image(test_image,use_column_width=True)
    #Predict Button
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Define Class
        class_name = ['bacterial_leaf_blight',
 'bacterial_leaf_streak',
 'brown_spot',
 'dead_heart',
 'hispa',
 'leaf_blast']
        st.success(f"Model is Predicting it's a {class_name[result_index]}")
