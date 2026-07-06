# ============================================================
#  TRAIN SIGN LANGUAGE MODEL WITH KAGGLE DATA - FIXED
# ============================================================

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import json
import sys

print("🚀 Training Sign Language Model")
print("=" * 50)

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Disable oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_sign_mnist():
    """Load Sign Language MNIST"""
    try:
        print("📂 Loading local files...")
        train = pd.read_csv('sign_mnist_train.csv')
        test = pd.read_csv('sign_mnist_test.csv')
        print("✅ Loaded from local files")
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("\nPlease download the data first:")
        print("  python download_data.py")
        sys.exit(1)
    
    # Separate labels and pixels
    y_train = train['label'].values
    x_train = train.drop('label', axis=1).values
    y_test = test['label'].values
    x_test = test.drop('label', axis=1).values
    
    print(f"📊 Original labels in training: {np.unique(y_train)}")
    print(f"📊 Number of original classes: {len(np.unique(y_train))}")
    
    # The dataset has classes 0-24 (25 classes)
    # Class 9 is 'J' - we want to remove it
    # Remap: 0-8 -> 0-8, 10-24 -> 9-23
    
    # Remove class 9 (J) from dataset
    train_mask = y_train != 9
    test_mask = y_test != 9
    
    y_train = y_train[train_mask]
    x_train = x_train[train_mask]
    y_test = y_test[test_mask]
    x_test = x_test[test_mask]
    
    # Create mapping for labels
    # Original: 0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24
    # New:      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23
    
    # Create a mapping dictionary
    label_map = {}
    new_label = 0
    for old_label in sorted(np.unique(y_train)):
        label_map[old_label] = new_label
        new_label += 1
    
    print(f"📊 Label mapping: {label_map}")
    
    # Apply mapping
    y_train = np.array([label_map[l] for l in y_train])
    y_test = np.array([label_map[l] for l in y_test])
    
    print(f"📊 Remapped labels: {np.unique(y_train)}")
    print(f"📊 Number of classes after remapping: {len(np.unique(y_train))}")
    
    # Reshape to 28x28 grayscale images
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    
    # One-hot encode labels (24 classes)
    num_classes = 24
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    print(f"✅ Loaded {len(x_train)} training, {len(x_test)} test samples")
    print(f"📊 Number of classes: {num_classes}")
    return x_train, y_train, x_test, y_test

def create_model():
    """Create CNN model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(24, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model():
    """Train the model"""
    print("\n📊 Loading data...")
    x_train, y_train, x_test, y_test = load_sign_mnist()
    
    print("\n🏗️ Building model...")
    model = create_model()
    model.summary()
    
    print("\n🚀 Training model (this may take 10-30 minutes)...")
    
    # Early stopping callback
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
    
    history = model.fit(
        x_train, y_train,
        epochs=10,  # Increase to 20-30 for better accuracy
        batch_size=64,
        validation_data=(x_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n✅ Test Accuracy: {test_acc:.4f}")
    
    # Create model directory
    os.makedirs('model', exist_ok=True)
    
    # Save model
    model.save('model/sign_model.h5')
    print("✅ Model saved as model/sign_model.h5")
    
    # Save labels (A-Z excluding J and Z)
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    with open('model/labels.json', 'w') as f:
        json.dump(labels, f)
    print("✅ Labels saved as model/labels.json")
    
    # Convert to TensorFlow.js
    print("\n🔄 Converting to TensorFlow.js format...")
    try:
        import tensorflowjs as tfjs
        os.makedirs('model/tfjs_model', exist_ok=True)
        tfjs.converters.save_keras_model(model, 'model/tfjs_model')
        print("✅ TensorFlow.js model saved to model/tfjs_model/")
        print("   Files:")
        for f in os.listdir('model/tfjs_model'):
            print(f"     - {f}")
    except ImportError:
        print("⚠️ tensorflowjs not installed. Run: pip install tensorflowjs")
    except Exception as e:
        print(f"⚠️ Error converting to TF.js: {e}")
    
    return model, history

if __name__ == '__main__':
    print("=" * 50)
    print("SIGNLENS AI - TRAINING PIPELINE")
    print("=" * 50)
    
    # Train model
    model, history = train_model()
    
    print("\n" + "=" * 50)
    print("✅ TRAINING COMPLETE!")
    print("📁 Model saved in 'model/' directory")
    print("🌐 Open index.html in your browser to test!")
    print("=" * 50)
