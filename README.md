# Face Shape and Hairstyle Recommendation App

The **Face Shape and Hairstyle Recommendation App** is a **Streamlit** web app that uses pre-trained deep learning models to predict face shapes and recommend suitable hairstyles based on the predicted face shape and gender. The app supports both image upload and webcam photo capture for face shape prediction. It also includes a login system, model performance metrics, and a section for common hair problems with tips and solutions.

## Key Features:
- **Face Shape Prediction**: Classifies face shapes into five categories: Heart, Oblong, Oval, Round, and Square.
- **Hairstyle Suggestions**: Provides personalized hairstyle recommendations based on the predicted face shape and gender.
- **Image Upload**: Users can upload an image of their face for prediction.
- **Webcam Prediction**: Take a live photo using your webcam for instant face shape prediction.
- **Login System**: The app is secured with a login page (default credentials: `Admin`/`123`).
- **Model Performance**: Displays model metrics like accuracy, loss, and confusion matrix for both CNN and VGG models.
- **Hair Care Tips**: Offers solutions for common hair problems like dandruff, hair fall, dry hair, oily hair, and split ends.

## Technologies Used:
- **Python**
- **Streamlit** for the frontend
- **TensorFlow** and **Keras** for model predictions
- **PIL** for image processing

## Installation & Setup:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/face-shape-hairstyle-app.git
   cd face-shape-hairstyle-app
