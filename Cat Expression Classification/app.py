import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# Fungsi untuk memuat model
def load_model():
    feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor_layer = hub.KerasLayer(feature_extractor_model, input_shape=(224, 224, 3), trainable=False)
    model = tf.keras.Sequential([feature_extractor_layer, tf.keras.layers.Dense(3, activation='softmax')])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load model saat aplikasi dimulai
model = load_model()

# Fungsi untuk melakukan prediksi
def predict_image(image):
    img_array = np.array(image.resize((224, 224)))
    img_array = img_array.astype('float32')  # Ubah tipe data menjadi float32
    img_array = np.expand_dims(img_array, axis=0)  # Tambah dimensi batch
    img_array /= 255.0  # Normalisasi

    
    # Prediksi menggunakan model
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    return predicted_class, predictions[0]

# Validasi input gambar
def is_valid_image(image, threshold=0.5):
    _, prediction_proba = predict_image(image)
    confidence = max(prediction_proba)
    return confidence >= threshold

content = '''
    <div style="text-align: center;">
        <h1>Cat Expression Recognition 😺😿😼</h1>
    </div>
    '''

# Setup Streamlit UI
st.markdown(content, unsafe_allow_html=True)
st.write("Upload an image of a cat to recognize its expression")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Tampilkan gambar yang diupload
    img = Image.open(uploaded_file)
    resized_img = img.resize((400, 400))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    if is_valid_image(img):
        predicted_class, prediction_proba = predict_image(img)
        class_labels = ['Angry', 'Sad', 'Happy']
        predicted_label = class_labels[predicted_class]
        
        st.write(f"Prediction: {predicted_label}")
        st.write(f"Prediction Probabilities: {prediction_proba}")
        
        # Visualisasi distribusi probabilitas
        fig, ax = plt.subplots()
        ax.bar(class_labels, prediction_proba, color=['red', 'blue', 'green'])
        ax.set_ylim([0, 1])
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probabilities')
        st.pyplot(fig)
    else:
        st.error("Invalid image: Please upload an image of a cat.")
