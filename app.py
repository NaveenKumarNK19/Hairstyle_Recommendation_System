import streamlit as st
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Set the page layout to wide
st.set_page_config(layout="wide")

# Define the username and password for the login
USERNAME = "Admin"
PASSWORD = "123"

# Cache model loading to speed up the app
@st.cache_resource()
def load_models():
    # Load the models (make sure to change the path if necessary)
    cnn_model = load_model('cnn_model.keras')
    vgg_model = load_model('vgg_model.keras')
    return cnn_model, vgg_model

# Load the models
cnn_model, vgg_model = load_models()

# Define the class labels (adjust according to your model's class labels)
class_labels = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

# Hairstyle suggestions based on face shape and gender
hairstyle_female = {
    'Heart': ['Side-Parted Long Waves', 'Textured Chin-Length Bob', 'Curtain Bangs with Long Layers'],
    'Oblong': ['Chin-Length Bob with Waves', 'Medium-Length Shag Cut', 'Layered Curls'],
    'Oval': ['Textured Lob (Long Bob)', 'Side-Swept Bangs with Long Layers', 'Shaggy Pixie Cut'],
    'Round': ['High-Volume Layered Bob', 'Asymmetrical Lob', 'Long, Loose Waves'],
    'Square': ['Soft Curled Bob', 'Layered Pixie Cut', 'Long, Side-Swept Waves']
}

hairstyle_male = {
    'Heart': ['Textured Side-Parted Waves', 'Messy Pompadour', 'Buzz Cut with Soft Fringe'],
    'Oblong': ['Choppy Textured Pompadour', 'Medium-Length Side Part', 'Loose Curls with Volume'],
    'Oval': ['Short Fade with Textured Top', 'Tapered Crew Cut', 'Classic Caesar Cut'],
    'Round': ['High-Top Fade', 'Undercut with Longer Top', 'Textured Quiff'],
    'Square': ['Buzz Cut', 'Layered Crop', 'Ivy League Cut']
}

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((150, 150))  # Resize to match input size of CNN model
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make predictions
def predict_image(model, img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_class_name = class_labels[predicted_class[0]]
    return predicted_class_name

# Function to display hairstyle recommendations
def show_hairstyles(predicted_class, gender):
    if gender == 'Female':
        recommended_hairstyles = hairstyle_female.get(predicted_class, [])
    else:
        recommended_hairstyles = hairstyle_male.get(predicted_class, [])
    
    if recommended_hairstyles:
        st.write("Recommended Hairstyles:")
        for hairstyle in recommended_hairstyles:
            st.write(f"- {hairstyle}")
    else:
        st.write("No hairstyle recommendations available for this face type.")

# Login page
def login():
    st.title("Login Page")
    
    # Centering the GIF using HTML
    st.markdown("""
        <style>
        .center {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        </style>
        <div class="center">
            <img src="https://cdn-icons-gif.flaticon.com/6569/6569164.gif" width="300" />
        </div>
    """, unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.session_state.logged_in = False
            st.error("Invalid username or password.")

# Main Streamlit app layout
if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    login()  # Display the login screen if the user is not logged in
else:
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ("Home", "Image Prediction", "Camera Prediction", "Hair Problems", "Results"))

    if page == "Home":
        st.title("Face Shape and Hairstyle Recommendation App")
        st.write("""
            Welcome to the Face Shape and Hairstyle Recommendation App!
            
            This app helps you identify your face shape and suggests suitable hairstyles based on the face shape.
            
            **How It Works:**
            1. Upload a photo of your face or use your webcam.
            2. The app will analyze the face shape (Heart, Oblong, Oval, Round, Square).
            3. Based on the predicted face shape and selected gender, the app will recommend hairstyles.
            
            **Models Used:**
            We use two Deep learning models:
            - A CNN-based model for face shape classification.
            - A VGG-based model for face shape classification.
            
            Choose your preferred model for predictions!
        """)

    elif page == "Image Prediction":
        st.title("Face Shape Prediction from Image")
        st.write("Upload an image to predict your face shape and get hairstyle suggestions.")
        
        # Dropdown to select gender
        gender = st.selectbox('Select Gender', ('Female', 'Male'))

        # Upload image
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        # Radio button to select the model (CNN or VGG)
        selected_model = st.radio("Select the model", ('CNN Model', 'VGG Model'))

        if uploaded_image is not None:
            # Open and display the uploaded image
            img = Image.open(uploaded_image)
            st.image(img, caption='Uploaded Image')

            if st.button("Predict"):
                if selected_model == 'CNN Model':
                    # Predict using CNN model
                    cnn_class = predict_image(cnn_model, img)
                    st.write(f"CNN Model Prediction: {cnn_class}")
                elif selected_model == 'VGG Model':
                    # Predict using VGG model
                    vgg_class = predict_image(vgg_model, img)
                    st.write(f"VGG Model Prediction: {vgg_class}")
                
                # Show hairstyle recommendations
                show_hairstyles(cnn_class if selected_model == 'CNN Model' else vgg_class, gender)

    elif page == "Camera Prediction":
        st.title("Face Shape Prediction from Camera")
        st.write("Use your webcam to take a photo and predict your face shape.")
        
        # Gender selection
        gender = st.selectbox('Select Gender', ('Female', 'Male'))

        # Use Streamlit's camera input for capturing image
        img = st.camera_input("Take a picture")

        # Radio button to select the model (CNN or VGG)
        selected_model = st.radio("Select the model", ('CNN Model', 'VGG Model'))

        if img is not None:
            # Open and display the captured image
            img = Image.open(img)
            st.image(img, caption='Captured Image')

            if st.button("Predict"):
                if selected_model == 'CNN Model':
                    # Predict using CNN model
                    cnn_class = predict_image(cnn_model, img)
                    st.write(f"CNN Model Prediction: {cnn_class}")
                elif selected_model == 'VGG Model':
                    # Predict using VGG model
                    vgg_class = predict_image(vgg_model, img)
                    st.write(f"VGG Model Prediction: {vgg_class}")
                
                # Show hairstyle recommendations
                show_hairstyles(cnn_class if selected_model == 'CNN Model' else vgg_class, gender)

    elif page == "Results":
        st.title("Model Results")
        st.write("""
            Here are the results for the trained models. The images below show:
            - CNN Accuracy
            - CNN Loss
            - CNN Confusion Matrix
            - VGG Accuracy
            - VGG Loss
            - VGG Confusion Matrix
        """)
        
        # Folder path where images are stored
        results_path = "RESULTS"
        
        # Make sure the folder exists
        if os.path.exists(results_path):
            # Define image paths
            cnn_accuracy_path = os.path.join(results_path, "cnn_accuracy.png")
            cnn_loss_path = os.path.join(results_path, "cnn_loss.png")
            cnn_cm_path = os.path.join(results_path, "cnn_cm.png")
            vgg_accuracy_path = os.path.join(results_path, "vgg_accuracy.png")
            vgg_loss_path = os.path.join(results_path, "vgg_loss.png")
            vgg_cm_path = os.path.join(results_path, "vgg_cm.png")

            # Check if the image files exist before displaying
            col1, col2, col3 = st.columns(3)
            with col1:
                if os.path.exists(cnn_accuracy_path):
                    st.image(cnn_accuracy_path, caption="CNN Accuracy")
                else:
                    st.write("CNN Accuracy Image not found.")

                if os.path.exists(vgg_accuracy_path):
                    st.image(vgg_accuracy_path, caption="VGG Accuracy")
                else:
                    st.write("VGG Accuracy Image not found.")

            with col2:
                if os.path.exists(cnn_loss_path):
                    st.image(cnn_loss_path, caption="CNN Loss")
                else:
                    st.write("CNN Loss Image not found.")

                if os.path.exists(vgg_loss_path):
                    st.image(vgg_loss_path, caption="VGG Loss")
                else:
                    st.write("VGG Loss Image not found.")

            with col3:
                if os.path.exists(cnn_cm_path):
                    st.image(cnn_cm_path, caption="CNN Confusion Matrix")
                else:
                    st.write("CNN Confusion Matrix Image not found.")
                    
                if os.path.exists(vgg_cm_path):
                    st.image(vgg_cm_path, caption="VGG Confusion Matrix")
                else:
                    st.write("VGG Confusion Matrix Image not found.")
                
        else:
            st.error("The RESULTS folder does not exist.")

    elif page == "Hair Problems":
        st.title("Common Hair Problems and Solutions")
        st.write("""
            Here are some common hair problems and possible solutions to manage them.
        """)

        # Create two columns
        col1, col2 = st.columns([1, 1])

        # Dandruff
        with col1:
            st.subheader("1. Dandruff")
            st.write("""
                Dandruff is a common condition of the scalp that causes flakes of dead skin to appear in the hair.
                - **Causes**: Dry skin, sensitivity to hair products, or fungal infections.
                - **Solutions**: 
                    - Use anti-dandruff shampoos.
                    - Moisturize your scalp with natural oils like coconut or tea tree oil.
                    - Avoid harsh hair care products.
            """)
        with col2:
            st.image("https://renu.doctor/wp-content/uploads/2023/10/Untitled-design-1024x576.jpg", caption="Dandruff", width=500)

        col1, col2 = st.columns([1, 1])

        # Hair Fall
        with col2:
            st.subheader("2. Hair Fall")
            st.write("""
                Hair fall can be caused by various factors such as stress, poor diet, or underlying medical conditions.
                - **Causes**: Stress, hormonal imbalance, poor diet, genetics.
                - **Solutions**:
                    - Eat a balanced diet rich in vitamins and minerals.
                    - Consider using hair growth serums.
                    - Manage stress through relaxation techniques.
            """)
        with col1:
            st.image("https://skinkraft.com/cdn/shop/articles/Hair-Loss_1024x1024.jpg?v=1584613267", caption="Hair Fall", width=500)

        col1, col2 = st.columns([1, 1])

        # Dry Hair
        with col1:
            st.subheader("3. Dry Hair")
            st.write("""
                Dry hair lacks moisture and can appear dull and frizzy.
                - **Causes**: Overuse of heat styling tools, frequent washing, or exposure to harsh weather.
                - **Solutions**: 
                    - Use moisturizing shampoos and conditioners.
                    - Limit heat styling and deep condition regularly.
            """)
        with col2:
            st.image("https://www.garnier.in/-/media/project/loreal/brand-sites/garnier/apac/in/all-article-pages/hair-care-tips/8-major-causes-of-dry-hair-that-you-need-to-know/banner-13.jpg?rev=3298d9e859bb45e8954fff76cdbb6adb&h=496&w=890&la=en-IN&hash=41FEA2DF6C9CBC2652BA425E4E51294A", caption="Dry Hair", width=500)

        col1, col2 = st.columns([1, 1])

        # Oily Hair
        with col2:
            st.subheader("4. Oily Hair")
            st.write("""
                Oily hair can feel greasy, especially at the roots, due to excess sebum production.
                - **Causes**: Overproduction of oil, hormonal changes, or a poor diet.
                - **Solutions**:
                    - Use a gentle shampoo for oily hair.
                    - Avoid washing hair too frequently to prevent overproduction of oil.
            """)
        with col1:
            st.image("https://cdn-blog.prose.com/1/2021/05/greasy-hair.jpg", caption="Oily Hair", width=500)

        col1, col2 = st.columns([1, 1])

        # Split Ends
        with col1:
            st.subheader("5. Split Ends")
            st.write("""
                Split ends occur when the hair shaft becomes frayed or damaged.
                - **Causes**: Lack of moisture, excessive heat styling, or chemical treatments.
                - **Solutions**:
                    - Trim your hair regularly.
                    - Use leave-in conditioners and hair oils to maintain moisture.
            """)
        with col2:
            st.image("https://www.fforhair.com/cdn/shop/articles/new-blog-landscape-1500-x-1000-px-1_4f9605a6-1bd5-4e02-944f-8db62a0a71fe.png?v=1726841245", caption="Split Ends", width=500)