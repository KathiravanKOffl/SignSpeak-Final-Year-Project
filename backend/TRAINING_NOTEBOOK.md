# SignSpeak Model Setup - Complete Guide

Three options to get a working model:
1. **Train from Scratch** (Custom data, takes 30 min)
2. **Download Pre-trained Model** (Instant, best for demo)
3. **Use WLASL Weights** (Pre-trained on video, needs adaptation)

---

## Option 1: Train from Scratch (Recommended for Learning)

### Cell 1: Upload Kaggle JSON & Install

```python
# Install dependencies
!pip install -q kagglehub torch numpy pandas scikit-learn tqdm mediapipe opencv-python

# Upload kaggle.json
from google.colab import files
print("📤 Upload your kaggle.json file...")
uploaded = files.upload()

# Setup credentials
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

import torch
print(f"✅ Setup complete! GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
# (No login needed since we used kaggle.json)
```

### Cell 2: Download & Train

```python
# Download public ASL Alphabet dataset
!kaggle datasets download -d grassknoted/asl-alphabet -p /content/asl_data --unzip

# (Paste the rest of the training code from previous version here...)
# ...
```

---

## Option 2: Download Community Pre-trained Model (Instant) 🚀

**Use this if you want to skip training!**

### Cell 1: Download Model

```python
import os
import urllib.request

print("📥 Downloading pre-trained ASL model...")

# Create checkpoint dir
os.makedirs('/content/SignSpeak-Final-Year-Project/backend/checkpoints', exist_ok=True)
dest_dir = '/content/SignSpeak-Final-Year-Project/backend/checkpoints'

# Download model from GitHub (rabBit64 - RNN/LSTM model)
# Source: https://github.com/rabBit64/Sign-language-recognition-with-RNN-and-Mediapipe
model_url = "https://github.com/rabBit64/Sign-language-recognition-with-RNN-and-Mediapipe/raw/main/action.h5"
urllib.request.urlretrieve(model_url, f"{dest_dir}/action.h5")

print("✅ Model downloaded: action.h5")

# Create label mapping for this specific model
# Based on repo documentation: ['hello', 'thanks', 'iloveyou']
import json
labels = ['hello', 'thanks', 'iloveyou'] 
mapping = {
    'id_to_label': {i: l for i, l in enumerate(labels)},
    'label_to_id': {l: i for i, l in enumerate(labels)}
}
with open(f"{dest_dir}/label_mapping.json", 'w') as f:
    json.dump(mapping, f)

print("✅ Label mapping created")
```

### Cell 2: Update Server for TensorFlow

You need to tell the server to load `.h5` instead of `.pth`.

```python
# Update server code to support .h5 (I will push this update to repo next)
```

---

## Option 3: Train on MediaPipe Dataset (High Accuracy)

### Cell 1: Download MediaPipe Feature Dataset

```python
import kagglehub
path = kagglehub.dataset_download("risangbaskoro/wlasl-mediapipe-features")
print(f"✅ Downloaded to: {path}")
```

### Cell 2: Train (Fast)

```python
# (Training code for feature CSVs...)
```
