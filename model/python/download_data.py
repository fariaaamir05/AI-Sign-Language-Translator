# ============================================================
#  DOWNLOAD SIGN LANGUAGE DATA FROM KAGGLE
# ============================================================

import os
import sys

print("📥 Downloading Sign Language MNIST from Kaggle...")
print("=" * 50)

try:
    # Try to import kagglehub
    import kagglehub
    print("✅ kagglehub found")
    
    # Download dataset
    print("📥 Downloading...")
    path = kagglehub.dataset_download("datamunge/sign-language-mnist")
    print(f"✅ Data downloaded to: {path}")
    
    # List files
    import shutil
    import pandas as pd
    
    os.makedirs('.', exist_ok=True)
    
    # Copy files to current directory
    files = os.listdir(path)
    for file in files:
        if file.endswith('.csv'):
            src = os.path.join(path, file)
            dst = os.path.join('.', file)
            shutil.copy(src, dst)
            print(f"   📄 Copied: {file}")
    
    print("\n✅ Data downloaded successfully!")
    print("📁 Files saved in current directory")
    
    # Show sample
    train = pd.read_csv('sign_mnist_train.csv')
    test = pd.read_csv('sign_mnist_test.csv')
    print(f"\n📊 Training samples: {len(train)}")
    print(f"📊 Test samples: {len(test)}")
    print(f"\n📊 Label distribution (first 10):")
    print(train['label'].value_counts().sort_index().head(10))
    
except ImportError:
    print("❌ kagglehub not installed.")
    print("\n📝 Install with: pip install kagglehub")
    print("\n📝 Or download manually from:")
    print("https://www.kaggle.com/datasets/datamunge/sign-language-mnist")
    print("\nDownload both CSV files and place them in this folder:")
    print("  - sign_mnist_train.csv")
    print("  - sign_mnist_test.csv")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("✅ Download complete!")
