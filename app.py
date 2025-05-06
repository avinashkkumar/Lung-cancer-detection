import os
import io
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the GradCAM class
from gradcam import GradCAM

# Create Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Model path
MODEL_PATH = 'models/lung_cancer_classifier.h5'

# Class names
CLASSES = ['Benign', 'Malignant', 'Normal']

# Global variables
model = None
gradcam = None
use_gradcam = True

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the trained model."""
    global model, gradcam, use_gradcam
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully!")
        
        # Initialize GradCAM
        try:
            gradcam = GradCAM(model)
            logger.info("GradCAM initialized successfully!")
            use_gradcam = True
        except Exception as e:
            logger.error(f"Error initializing GradCAM: {e}")
            logger.info("Continuing without GradCAM visualization")
            use_gradcam = False
        
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess the image for prediction."""
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize
    img_array = np.array(img) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img, img_array

def generate_fallback_visualization(original_img):
    """Generate a fallback visualization when GradCAM fails."""
    # Create a simple colored border as a fallback visualization
    img_with_border = original_img.copy()
    h, w, _ = img_with_border.shape
    border_size = int(min(h, w) * 0.05)  # 5% of the smallest dimension
    
    # Add a colored border (blue)
    img_with_border = cv2.copyMakeBorder(
        img_with_border, 
        border_size, border_size, border_size, border_size, 
        cv2.BORDER_CONSTANT, 
        value=[0, 0, 255]
    )
    
    return img_with_border

def predict_image(image_path):
    """Make a prediction for an image."""
    global use_gradcam
    
    # Preprocess image
    original_img, img_array = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    prediction_confidence = float(predictions[0][predicted_class_idx])
    predicted_class = CLASSES[predicted_class_idx]
    
    # Generate visualization
    if use_gradcam and gradcam is not None:
        try:
            # Try to generate GradCAM visualization
            visualization = gradcam.explain(original_img, class_idx=predicted_class_idx, alpha=0.7)
            
            # Enhance visualization colors if needed
            if np.mean(np.abs(np.diff(visualization, axis=0))) < 10:
                # If the visualization doesn't have much variation, enhance it
                logger.info("Enhancing visualization contrast")
                visualization = enhance_visualization(visualization)
                
        except Exception as e:
            logger.error(f"Error generating GradCAM visualization: {e}")
            # Fallback to a simpler visualization
            visualization = generate_fallback_visualization(original_img)
            use_gradcam = False  # Disable for future requests
    else:
        # Use fallback visualization if GradCAM is not available
        visualization = generate_fallback_visualization(original_img)
    
    # Convert visualization to base64 for web display
    _, buffer = cv2.imencode('.png', cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    # Create class probabilities
    class_probabilities = {cls: float(prob) for cls, prob in zip(CLASSES, predictions[0])}
    
    return {
        'prediction': predicted_class,
        'confidence': prediction_confidence,
        'class_probabilities': class_probabilities,
        'heatmap_image': img_str,
        'grad_cam_enabled': use_gradcam and gradcam is not None
    }

def enhance_visualization(image):
    """Enhance the visualization to make colors more vibrant."""
    # Convert to HSV for better color manipulation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Increase saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255).astype(np.uint8)
    
    # Increase value/brightness for more vibrant colors
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255).astype(np.uint8)
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return enhanced

# Routes
@app.route('/')
def home():
    """Home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Process the uploaded image and return prediction."""
    # Check if model is loaded
    global model
    if model is None:
        if not load_model():
            return jsonify({'error': 'Failed to load model. Please try again later.'}), 500
    
    # Check if image was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file is allowed
    if file and allowed_file(file.filename):
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Make prediction
        try:
            result = predict_image(file_path)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return jsonify({'error': f'Error during prediction: {str(e)}'}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/predict-api', methods=['POST'])
def predict_api():
    """API endpoint for prediction."""
    # Check if model is loaded
    global model
    if model is None:
        if not load_model():
            return jsonify({'error': 'Failed to load model. Please try again later.'}), 500
    
    # Get image from request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # Check if file is allowed
    if file and allowed_file(file.filename):
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Make prediction
        try:
            result = predict_image(file_path)
            
            # Clean up file if needed
            # os.remove(file_path)
            
            return jsonify({
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'class_probabilities': result['class_probabilities']
            })
        except Exception as e:
            logger.error(f"Error during API prediction: {str(e)}")
            return jsonify({'error': f'Error during prediction: {str(e)}'}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    # Load model when app starts
    load_model()
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True) 