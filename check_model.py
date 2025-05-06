import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_model_setup():
    """Check if the model directory and file exist."""
    # Define paths
    MODEL_DIR = 'models'
    MODEL_PATH = os.path.join(MODEL_DIR, 'lung_cancer_classifier.h5')
    
    # Check if model directory exists
    if not os.path.exists(MODEL_DIR):
        logger.info(f"Creating model directory: {MODEL_DIR}")
        os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}")
        logger.info("Please train the model first by running: python train_model.py")
        return False
    
    # Check model file size
    file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # Size in MB
    logger.info(f"Model file found: {MODEL_PATH} ({file_size:.2f} MB)")
    
    # Check if uploads directory exists
    UPLOAD_DIR = 'uploads'
    if not os.path.exists(UPLOAD_DIR):
        logger.info(f"Creating upload directory: {UPLOAD_DIR}")
        os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # Check if processed_dataset directory exists
    DATASET_DIR = 'processed_dataset'
    if not os.path.exists(DATASET_DIR):
        logger.warning(f"Processed dataset directory not found: {DATASET_DIR}")
        logger.info("You may need to run: python prepare_data.py")
    
    return True

if __name__ == "__main__":
    logger.info("Checking model setup...")
    result = check_model_setup()
    
    if result:
        logger.info("Setup check passed. You can now run the app with: python app.py")
        sys.exit(0)
    else:
        logger.error("Setup check failed. Please address the issues above.")
        sys.exit(1) 