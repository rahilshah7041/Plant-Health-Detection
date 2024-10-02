import pickle
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
model = load_model('C:\\Users\\Manush Khatri\\Desktop\\PHD\\PlantVillage\\PDD_completemodelstar.h5')

# Load plant disease labels
filename = 'plant_disease_label_transform.pkl'
image_labels = pickle.load(open(filename, 'rb'))

# Dimensions of resized image
DEFAULT_IMAGE_SIZE = tuple((256, 256))

def convert_image_to_array(img):
    try:
        # Convert image to array
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize the pixel values
        return img_array
    except Exception as e:
        print(f"Error : {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'})
        
        # Get the uploaded image file
        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'No file selected'})

        # Read the image file
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, DEFAULT_IMAGE_SIZE)

        # Make predictions
        img_array = convert_image_to_array(img)
        if img_array is None:
            return jsonify({'error': 'Failed to convert image to array'})

        print("Image Shape:", img_array.shape)

        predictions = model.predict(img_array)

        # Get the index of the class with the highest probability
        predicted_class_index = np.argmax(predictions)

        # Get the corresponding class label
        predicted_class_label = image_labels.classes_[predicted_class_index]

        return jsonify({'predicted_class': predicted_class_label})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

