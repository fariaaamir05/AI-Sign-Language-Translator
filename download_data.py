# ============================================================
#  DOWNLOAD SIGN LANGUAGE DATA FROM KAGGLE
# ============================================================

import os
import sys
import shutil

print("📥 Downloading Sign Language MNIST from Kaggle...")
print("=" * 50)

try:
    import kagglehub
    print("✅ kagglehub found")
    
    # Download dataset
    print("📥 Downloading...")
    path = kagglehub.dataset_download("datamunge/sign-language-mnist")
    print(f"✅ Data downloaded to: {path}")
    
    # Copy files to current directory
    files = os.listdir(path)
    csv_files = [f for f in files if f.endswith('.csv')]
    
    if not csv_files:
        print("⚠️ No CSV files found in download!")
        print(f"📁 Contents: {files}")
        sys.exit(1)
    
    for file in csv_files:
        src = os.path.join(path, file)
        dst = os.path.join('.', file)
        shutil.copy(src, dst)
        print(f"   📄 Copied: {file}")
    
    print("\n✅ Data downloaded successfully!")
    
    # Verify files
    import pandas as pd
    if os.path.exists('sign_mnist_train.csv'):
        train = pd.read_csv('sign_mnist_train.csv')
        print(f"📊 Training samples: {len(train)}")
    if os.path.exists('sign_mnist_test.csv'):
        test = pd.read_csv('sign_mnist_test.csv')
        print(f"📊 Test samples: {len(test)}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\n📝 Install with: pip install kagglehub")
    print("\n📝 Or download manually from:")
    print("https://www.kaggle.com/datasets/datamunge/sign-language-mnist")
    print("\nDownload both CSV files and place them in this folder:")
    print("  - sign_mnist_train.csv")
    print("  - sign_mnist_test.csv")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n📝 Try downloading manually from:")
    print("https://www.kaggle.com/datasets/datamunge/sign-language-mnist")
    sys.exit(1)

print("\n" + "=" * 50)
print("✅ Download complete!")
