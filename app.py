import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from breast_cancer_detection import create_model, preprocess_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
model = create_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the image
            features, processed_path = preprocess_image(filepath)
            
            # Make prediction
            prediction = model.predict(features)
            probability = model.predict_proba(features)[0]
            
            # Get the result
            result = "Malignant" if prediction[0] == 1 else "Benign"
            confidence = float(probability[1] if prediction[0] == 1 else probability[0])
            
            return jsonify({
                'result': result,
                'confidence': confidence,
                'image_path': filepath,
                'processed_image_path': processed_path
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True) 