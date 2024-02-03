import streamlit as st
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO

# Function to preprocess the image
def preprocess_image(image_path):
    pil_image = Image.open(image_path)
    pil_image_resized = pil_image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(pil_image_resized)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array /= 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Main Streamlit code
st.title("Dog vs Cat Image Classifier")

# Upload an image through Streamlit
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

# Load the pre-trained model only if an image is uploaded
model_url = 'https://raw.githubusercontent.com/Smail-Bel/test_1/main/dog_cat_detector_model_Final_2.h5'

try:
    # Fetch the model content from GitHub
    model_content = requests.get(model_url).content

    # Load the pre-trained model from in-memory content
    model = tf.keras.models.load_model(BytesIO(model_content))

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            pil_image = Image.open(uploaded_file)
            st.image(pil_image, caption="Uploaded Image", use_column_width=True)

            # Preprocess the resized image for the model
            img_array = preprocess_image(uploaded_file)

            # Make predictions using the loaded model
            predictions = model.predict(img_array)

            # Print raw predictions
            st.write("Raw Predictions:", predictions[0].tolist())

            # Manually interpret predictions based on your model's output with a threshold
            threshold = 0.5
            predicted_class = 'Dog' if predictions[0][0] >= threshold else 'Cat'
            confidence = predictions[0][0]

            # Display the result
            st.subheader("Prediction:")
            st.write(f"Predicted class: {predicted_class}")
            st.write(f"Confidence: {float(confidence):.2%}")

        except Image.UnidentifiedImageError as e:
            st.error(f"Error: Unable to identify image file. Please upload a valid image.")
        except Exception as e:
            st.error(f"Error processing the image: {e}")

except requests.exceptions.RequestException as e:
    st.error(f"Error fetching model from GitHub: {e}")

except Exception as e:
    st.error(f"Error loading the model: {e}")
