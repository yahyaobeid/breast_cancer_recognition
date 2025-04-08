import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BreastCancerModel:
    def __init__(self):
        self.model = None
        self.image_size = (224, 224)
        self.batch_size = 16
        self.epochs = 30
        self.dataset_dir = "dataset"
        self.model_path = "saved_model"
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        # Try to load existing model
        self._load_model()
        
    def _load_model(self):
        """Load the model if it exists."""
        try:
            if os.path.exists(os.path.join(self.model_path, "model.h5")):
                self.model = tf.keras.models.load_model(os.path.join(self.model_path, "model.h5"))
                logger.info("Loaded existing model")
                return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
        return False
    
    def _save_model(self):
        """Save the trained model."""
        try:
            if self.model is not None:
                self.model.save(os.path.join(self.model_path, "model.h5"))
                logger.info("Model saved successfully")
                return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
        return False
    
    def _create_model(self):
        """Create a CNN model for breast cancer detection."""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(64, (3, 3), padding='same', input_shape=(224, 224, 3)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(256, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(512, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def _create_data_augmentation(self):
        """Create data augmentation pipeline."""
        return tf.keras.Sequential([
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomFlip("horizontal"),
            layers.RandomBrightness(0.2),
            layers.RandomContrast(0.2),
        ])
    
    def _load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image."""
        try:
            # Load image
            img = load_img(image_path, target_size=self.image_size)
            # Convert to array
            img_array = img_to_array(img)
            # Normalize pixel values
            img_array = img_array.astype('float32') / 255.0
            return img_array
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def _load_dataset(self, subset='train'):
        """Load images from the dataset directory."""
        images = []
        labels = []
        
        # Load benign cases
        benign_dir = os.path.join(self.dataset_dir, subset, 'benign')
        for img_name in os.listdir(benign_dir):
            if img_name.endswith('.png'):
                img_path = os.path.join(benign_dir, img_name)
                img_array = self._load_and_preprocess_image(img_path)
                if img_array is not None:
                    images.append(img_array)
                    labels.append(0)  # 0 for benign
        
        # Load malignant cases
        malignant_dir = os.path.join(self.dataset_dir, subset, 'malignant')
        for img_name in os.listdir(malignant_dir):
            if img_name.endswith('.png'):
                img_path = os.path.join(malignant_dir, img_name)
                img_array = self._load_and_preprocess_image(img_path)
                if img_array is not None:
                    images.append(img_array)
                    labels.append(1)  # 1 for malignant
        
        return np.array(images), np.array(labels)
    
    def train(self):
        """Train the model on the organized dataset."""
        try:
            # Load training data
            logger.info("Loading training data...")
            X_train, y_train = self._load_dataset('train')
            logger.info(f"Loaded {len(X_train)} training images")
            
            # Load validation data
            logger.info("Loading validation data...")
            X_val, y_val = self._load_dataset('val')
            logger.info(f"Loaded {len(X_val)} validation images")
            
            # Print class distribution
            train_dist = np.bincount(y_train)
            val_dist = np.bincount(y_val)
            logger.info(f"Training set distribution - Benign: {train_dist[0]}, Malignant: {train_dist[1]}")
            logger.info(f"Validation set distribution - Benign: {val_dist[0]}, Malignant: {val_dist[1]}")
            
            # Calculate class weights
            total = len(y_train)
            weight_for_0 = (1 / train_dist[0]) * (total / 2.0)
            weight_for_1 = (1 / train_dist[1]) * (total / 2.0)
            class_weights = {0: weight_for_0, 1: weight_for_1}
            
            # Create data augmentation
            data_augmentation = self._create_data_augmentation()
            
            # Create and train model
            logger.info("Creating model...")
            self.model = self._create_model()
            
            # Create callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3,
                    min_lr=1e-6
                )
            ]
            
            logger.info("Starting training...")
            history = self.model.fit(
                data_augmentation(X_train),
                y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(X_val, y_val),
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save the trained model
            self._save_model()
            
            logger.info("Training completed successfully!")
            return history
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def predict(self, image_path):
        """Make a prediction on a single image."""
        try:
            if self.model is None:
                logger.error("Model not trained yet!")
                return None, None
            
            # Load and preprocess the image
            img_array = self._load_and_preprocess_image(image_path)
            if img_array is None:
                return None, None
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            prediction = self.model.predict(img_array, verbose=0)[0][0]
            
            # Convert to class and confidence
            predicted_class = "Malignant" if prediction > 0.5 else "Benign"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            return predicted_class, confidence
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return None, None

if __name__ == "__main__":
    # Train the model
    model = BreastCancerModel()
    if not model.model:  # Only train if no model exists
        logger.info("No existing model found. Starting training...")
        model.train()
    else:
        logger.info("Using existing trained model") 