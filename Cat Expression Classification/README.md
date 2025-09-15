# Cat Expression Classifier

This repository contains the implementation and deployment of a machine learning model that classifies cat expressions into multiple categories. The model leverages a pre-trained MobileNetV2 architecture with fine-tuning for enhanced accuracy and is deployed using Streamlit for a user-friendly web interface. for the .ipynb file there is a deep-learning-projects repository

---

## Features
- **Pre-trained Model**: Utilizes MobileNetV2 from TensorFlow Hub for transfer learning and fine-tuning.
- **User Interaction**: Upload an image directly through the interface.
- **Real-time Predictions**: Provides immediate classification results.
- **Responsive Display**: Ensures consistent image display size for a polished user experience.

---

## Architecture

The architecture used in this project is as follows:

1. **Pre-trained MobileNetV2 Feature Extractor**:
   - The feature extraction layer from MobileNetV2 (available on TensorFlow Hub) was used to leverage pre-trained knowledge on the ImageNet dataset.
   - Trainable layers were enabled during fine-tuning to adapt the model to the specific cat expression dataset.

2. **Custom Classification Head**:
   - Added a dense layer to map the extracted features to the specific classes of cat expressions.
   - Used a softmax activation function for multi-class classification.

3. **Data Preprocessing**:
   - Images were resized to 224x224 pixels.
   - Pixel values normalized to the range [0, 1].

4. **Deployment**:
   - The trained model was deployed using Streamlit, providing an intuitive interface for users to upload images and view predictions.

---

## Installation

Follow these steps to set up the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cat-expression-classifier.git
   cd cat-expression-classifier
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## Usage

1. Open the Streamlit app in your browser (usually at `http://localhost:8501`).
2. Upload an image of a cat using the file uploader.
3. View the predicted class and the uploaded image on the interface.

---

## Example

### Input:
An image of a cat expressing curiosity.

### Output:
- **Prediction**: "Curious"
- **Confidence**: 91.5%

---

## Results

The model achieved the following performance metrics:
- **Precision**: 92%
- **Recall**: 90%
- **F1-Score**: 90%

---

## Contributing

Feel free to submit issues or pull requests if you want to contribute to this project.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgements

- TensorFlow Hub for the pre-trained MobileNetV2 model.
- Streamlit for providing an easy-to-use deployment platform.

---

Enjoy classifying cat expressions!

You can try it self by this link:
https://cat-expression-classification.streamlit.app/
