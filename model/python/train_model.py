# ============================================================
#  TRAIN SIGN LANGUAGE MODEL WITH KAGGLE DATA
#  Run this on Kaggle or locally with downloaded dataset
# ============================================================

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import json
import sys
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("🚀 Training Sign Language Model with Kaggle Data")
print("=" * 50)

# ===== CHECK TENSORFLOW VERSION =====
print(f"TensorFlow version: {tf.__version__}")

# ===== LOAD DATA =====
def load_sign_mnist():
    """Load Sign Language MNIST from Kaggle"""
    try:
        # Try to load from local files first
        print("📂 Attempting to load local files...")
        train = pd.read_csv('sign_mnist_train.csv')
        test = pd.read_csv('sign_mnist_test.csv')
        print("✅ Loaded from local files")
    except FileNotFoundError:
        # Try to download from Kaggle
        print("📥 Local files not found. Trying to download from Kaggle...")
        try:
            import kagglehub
            path = kagglehub.dataset_download("datamunge/sign-language-mnist")
            train = pd.read_csv(f"{path}/sign_mnist_train.csv")
            test = pd.read_csv(f"{path}/sign_mnist_test.csv")
            print("✅ Downloaded from Kaggle")
        except ImportError:
            print("❌ kagglehub not installed. Install with: pip install kagglehub")
            print("📝 Or download manually from: https://www.kaggle.com/datasets/datamunge/sign-language-mnist")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error downloading: {e}")
            sys.exit(1)
    
    # Separate labels and pixels
    y_train = train['label'].values
    x_train = train.drop('label', axis=1).values
    y_test = test['label'].values
    x_test = test.drop('label', axis=1).values
    
    # Reshape to 28x28 grayscale images
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    
    # One-hot encode labels (24 classes: A-Z excluding J, Z)
    num_classes = 24
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    print(f"✅ Loaded {len(x_train)} training, {len(x_test)} test samples")
    print(f"📊 Number of classes: {num_classes}")
    return x_train, y_train, x_test, y_test

# ===== BUILD MODEL =====
def create_model():
    """Create CNN model for sign language recognition"""
    model = models.Sequential([
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(24, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ===== TRAIN =====
def train_model():
    """Train the model"""
    print("\n📊 Loading data...")
    x_train, y_train, x_test, y_test = load_sign_mnist()
    
    print("\n🏗️ Building model...")
    model = create_model()
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.00001)
    ]
    
    print("\n🚀 Training model...")
    history = model.fit(
        x_train, y_train,
        epochs=30,
        batch_size=64,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n✅ Test Accuracy: {test_acc:.4f}")
    
    # Create model directory
    os.makedirs('model', exist_ok=True)
    
    # Save in Keras format
    model.save('model/sign_model.h5')
    print("✅ Model saved as model/sign_model.h5")
    
    # Save labels
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    with open('model/labels.json', 'w') as f:
        json.dump(labels, f)
    print("✅ Labels saved as model/labels.json")
    
    # ===== CONVERT TO TENSORFLOW.JS =====
    print("\n🔄 Converting to TensorFlow.js format...")
    try:
        import tensorflowjs as tfjs
        # Create tfjs model directory
        os.makedirs('model/tfjs_model', exist_ok=True)
        tfjs.converters.save_keras_model(model, 'model/tfjs_model')
        print("✅ TensorFlow.js model saved to model/tfjs_model/")
    except ImportError:
        print("⚠️ tensorflowjs not installed. Run: pip install tensorflowjs")
        print("   Model will still work in Python but not in the browser")
    
    return model, history

# ===== EXPORT METADATA =====
def export_model_metadata():
    """Create metadata file for web deployment"""
    metadata = {
        "name": "SignLens AI Model",
        "version": "1.0",
        "source": "Kaggle Sign Language MNIST",
        "classes": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'],
        "input_shape": [28, 28, 1],
        "num_classes": 24,
        "accuracy": None  # Will be updated after training
    }
    with open('model/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✅ Metadata saved to model/metadata.json")

# ===== MAIN =====
if __name__ == '__main__':
    print("=" * 50)
    print("SIGNLENS AI - TRAINING PIPELINE")
    print("=" * 50)
    
    # Train model
    model, history = train_model()
    
    # Export metadata
    export_model_metadata()
    
    print("\n" + "=" * 50)
    print("✅ TRAINING COMPLETE!")
    print("📁 Model saved in 'model/' directory")
    print("🌐 Ready for TensorFlow.js deployment")
    print("=" * 50)
    
    # Print summary
    final_accuracy = history.history['val_accuracy'][-1]
    print(f"\n📊 Final Validation Accuracy: {final_accuracy:.4f}")