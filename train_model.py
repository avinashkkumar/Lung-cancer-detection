import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import get_custom_objects
import time

# Set constants
PROCESSED_DATASET = 'processed_dataset'
TRAIN_DIR = os.path.join(PROCESSED_DATASET, 'train')
VAL_DIR = os.path.join(PROCESSED_DATASET, 'val')
TEST_DIR = os.path.join(PROCESSED_DATASET, 'test')
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'lung_cancer_classifier.h5')
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50
NUM_CLASSES = 3
CLASSES = ['Benign', 'Malignant', 'Normal']

# Create model directory
os.makedirs(MODEL_DIR, exist_ok=True)

def create_data_generators():
    """Create training, validation and test data generators."""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescale for validation and test
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def create_model():
    """Create a transfer learning model using MobileNetV2."""
    # Load the pretrained model
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Create and compile the model
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def unfreeze_model_layers(model, base_model, train_generator):
    """Unfreeze some layers of the base model for fine-tuning."""
    # Unfreeze the last few layers
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_generator, val_generator):
    """Train the model."""
    # Define callbacks
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Train the model
    print("Training the model (initial phase)...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=15,
        callbacks=callbacks
    )
    
    return model, history

def fine_tune_model(model, base_model, train_generator, val_generator):
    """Fine-tune the model by unfreezing some layers."""
    # Unfreeze layers
    model = unfreeze_model_layers(model, base_model, train_generator)
    
    # Define callbacks for fine-tuning
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Fine-tune the model
    print("Fine-tuning the model...")
    fine_tune_history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    return model, fine_tune_history

def evaluate_model(model, test_generator):
    """Evaluate the model on the test set."""
    # Get the true labels
    y_true = test_generator.classes
    
    # Make predictions
    y_pred_prob = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    print(report)
    
    # Compute and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES, rotation=45)
    plt.yticks(tick_marks, CLASSES)
    
    # Add text annotations to the confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
    plt.close()
    
    # Return accuracy score
    accuracy = np.sum(y_pred == y_true) / len(y_true)
    return accuracy, report

def plot_training_history(history, fine_tune_history=None):
    """Plot training and validation accuracy/loss curves."""
    # Combine histories if fine-tuning was done
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    if fine_tune_history:
        acc += fine_tune_history.history['accuracy']
        val_acc += fine_tune_history.history['val_accuracy']
        loss += fine_tune_history.history['loss']
        val_loss += fine_tune_history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
    plt.close()

def main():
    """Main function to train and evaluate the model."""
    print("Starting model training...")
    start_time = time.time()
    
    # Create data generators
    train_generator, val_generator, test_generator = create_data_generators()
    
    # Create model
    model, base_model = create_model()
    print(model.summary())
    
    # Train model (initial phase)
    model, history = train_model(model, train_generator, val_generator)
    
    # Fine-tune model
    model, fine_tune_history = fine_tune_model(model, base_model, train_generator, val_generator)
    
    # Plot training history
    plot_training_history(history, fine_tune_history)
    
    # Evaluate model
    accuracy, _ = evaluate_model(model, test_generator)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    
    end_time = time.time()
    print(f"Total training time: {(end_time - start_time) / 60:.2f} minutes")
    print("Model training and evaluation completed!")

if __name__ == "__main__":
    main() 