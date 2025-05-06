# Lung Cancer CT Scan Classification

A deep learning system to classify lung CT scan images as Normal, Benign, or Malignant, with explanation capabilities.

## Project Overview

This project uses deep learning to analyze lung CT scan images and identify signs of cancer. It features:

- A TensorFlow-based CNN model for classification
- Transfer learning with MobileNetV2 architecture 
- Model explainability using Grad-CAM
- Web interface for uploading and analyzing images

## Dataset

The project uses the IQ-OTHNCCD lung cancer dataset that includes:
- Benign cases: Lung abnormalities that are not cancerous
- Malignant cases: Confirmed lung cancer
- Normal cases: Healthy lung tissue

## Features

1. **Image Classification**: Categorizes CT scan images into Normal, Benign, or Malignant
2. **Confidence Scores**: Provides probability for each class
3. **Visual Explanation**: Highlights areas that influenced the model's decision
4. **User-friendly Interface**: Web interface for easy image upload and result viewing
5. **API Support**: REST API for integration with other systems

## Project Files

- `prepare_data.py`: Script to organize dataset into train/validation/test splits
- `train_model.py`: Model training and evaluation script
- `gradcam.py`: Implementation of Grad-CAM for model explainability
- `app.py`: Flask web server for hosting the prediction service
- `templates/index.html`: Web interface for image upload and analysis

## Setup and Installation

### Prerequisites

- Python 3.7+
- TensorFlow 2.12+
- Flask 3.0+
- OpenCV
- Other dependencies listed in requirements.txt

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd lung-cancer-classifier
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare the dataset:
   ```
   python prepare_data.py
   ```

4. Train the model:
   ```
   python train_model.py
   ```

5. Run the web server:
   ```
   python app.py
   ```

## Troubleshooting

If you encounter any issues, try these troubleshooting steps:

1. **Model not found error**:
   - Make sure you've run `python train_model.py` before starting the app
   - Verify the model file exists at `models/lung_cancer_classifier.h5`
   - Run `python check_model.py` to verify your setup

2. **GradCAM visualization issues**:
   - The app will fall back to a simplified visualization if GradCAM fails
   - This does not affect the classification accuracy, only the explanation visualization
   - Check the logs for specific error messages

3. **Server errors (500)**:
   - Check the console/terminal where the Flask app is running for detailed error messages
   - Ensure all dependencies are installed with `pip install -r requirements.txt`
   - Verify that the image format is supported (JPG, JPEG, PNG)

4. **Training issues**:
   - Ensure dataset is properly structured by running `python prepare_data.py`
   - Check for any errors during the training process
   - Try reducing batch size if you encounter memory issues

5. **Other issues**:
   - Clear your browser cache
   - Restart the Flask server
   - Check the logs for additional information

## Usage

### Model Training

To train the model with custom parameters:

```python
python train_model.py
```

The training script:
- Uses transfer learning with MobileNetV2
- Applies data augmentation to improve generalization
- Implements fine-tuning after initial training
- Saves the best model based on validation accuracy

### Web Interface

1. Start the Flask server:
   ```
   python app.py
   ```

2. Open a web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Upload a CT scan image using the interface
4. View classification results and explanation

### API Usage

You can also use the REST API directly:

```bash
curl -X POST -F "file=@/path/to/your/image.jpg" http://localhost:5000/predict-api
```

## Model Performance

The model achieves robust performance metrics:
- Multi-class classification using MobileNetV2
- Optimized for medical imaging with domain-specific hyperparameters
- Explainability through Grad-CAM visualizations

## License

[License information here]

## Acknowledgments

Based on the IQ-OTHNCCD lung cancer dataset from the Iraq-Oncology Teaching Hospital/National Center for Cancer Diseases. 