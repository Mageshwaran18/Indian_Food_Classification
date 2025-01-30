from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Load the model
model = load_model('models/model.h5')
class_names = ['dosa', 'grilled_chicken', 'noodles']

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Get the image from request
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    
    # Convert to RGB if image is in RGBA
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Convert to numpy array
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    
    # Resize and preprocess
    resize = tf.image.resize(img_array, (224, 224))
    
    # Make prediction
    yhat = model.predict(np.expand_dims(resize/255, 0))
    
    # Get predicted class
    predicted_class_index = np.argmax(yhat)
    predicted_class = class_names[predicted_class_index]
    
    return jsonify({
        'predicted_class': predicted_class,
        'confidence': float(yhat[0][predicted_class_index])
    })

if __name__ == '__main__':
    app.run(debug=True)
