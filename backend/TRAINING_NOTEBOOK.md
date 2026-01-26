# SignSpeak Training Notebook - Final Working Version (Option 2)

**Goal:** Deployment without dataset headaches.
**Strategy:** Download a working pre-trained model directly from GitHub.

---

## 🟢 Cell 1: Setup
Install dependencies.

```python
!pip install -q torch numpy pandas scikit-learn tqdm

import torch
import os
import urllib.request
import json
print(f"✅ Setup complete! GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## 🟢 Cell 2: Download Pre-trained Model (Direct)
We will use a community model (rabBit64) trained on MediaPipe landmarks.

```python
print("📥 Downloading pre-trained model...")

# Create checkpoint dir
os.makedirs('checkpoints', exist_ok=True)

# Correct raw URL for the model
model_url = "https://github.com/rabBit64/Sign-language-recognition-with-RNN-and-Mediapipe/raw/main/action.h5"
try:
    urllib.request.urlretrieve(model_url, "checkpoints/action.h5")
    print("✅ Model 'action.h5' downloaded!")
except:
    # Fallback name if action.h5 doesn't exist
    model_url = "https://github.com/rabBit64/Sign-language-recognition-with-RNN-and-Mediapipe/raw/main/model.h5"
    urllib.request.urlretrieve(model_url, "checkpoints/best_model.h5")
    print("✅ Model 'best_model.h5' downloaded!")

# Create corresponding label mapping
# Based on rabBit64 repo: ['hello', 'thanks', 'iloveyou']
labels = ['hello', 'thanks', 'iloveyou'] 
mapping = {
    'id_to_label': {i: l for i, l in enumerate(labels)},
    'label_to_id': {l: i for i, l in enumerate(labels)}
}
with open("checkpoints/label_mapping.json", 'w') as f:
    json.dump(mapping, f)

print("✅ Label mapping created")
```

---

## 🟢 Cell 3: Deploy to Backend
Updates your backend to use this new model.

```python
import shutil

# 1. Update repo (clean start)
if os.path.exists('/content/SignSpeak-Final-Year-Project'):
    !rm -rf /content/SignSpeak-Final-Year-Project
!git clone https://github.com/KathiravanKOffl/SignSpeak-Final-Year-Project.git

# 2. Copy model
dest = '/content/SignSpeak-Final-Year-Project/backend/checkpoints/'
os.makedirs(dest, exist_ok=True)

if os.path.exists("checkpoints/action.h5"):
    shutil.copy("checkpoints/action.h5", dest + "best_model.h5")
elif os.path.exists("checkpoints/best_model.h5"):
    shutil.copy("checkpoints/best_model.h5", dest + "best_model.h5")
    
shutil.copy("checkpoints/label_mapping.json", dest)
print(f"✅ Model deployed to: {dest}")

# 3. Start Server
print("\n🚀 RESTARTING SERVER...")
%cd /content/SignSpeak-Final-Year-Project/backend
!python -m uvicorn api.inference_server_wlasl:app --host 0.0.0.0 --port 8000
```
