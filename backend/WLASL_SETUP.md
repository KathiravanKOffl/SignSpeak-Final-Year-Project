# Pre-Trained ASL Model Setup

Quick Colab cells to download and integrate WLASL pre-trained model.

---

## Cell 1: Download Pre-Trained Pose-TGCN Model

```python
import os
import gdown

print("📥 Downloading pre-trained Pose-TGCN model...")

# Create directories
os.makedirs('/content/wlasl_weights', exist_ok=True)

# Pose-TGCN pre-trained weights
# Source: https://drive.google.com/file/d/1dzvocsaylRsjqaY4r_lyRihPZn0I6AA_/view
gdown.download(
    'https://drive.google.com/uc?id=1dzvocsaylRsjqaY4r_lyRihPZn0I6AA_', 
    '/content/wlasl_weights/pose_tgcn.zip',
    quiet=False
)

!unzip -o /content/wlasl_weights/pose_tgcn.zip -d /content/wlasl_weights/

print("✅ Pre-trained weights downloaded!")
print("📂 Contents:", os.listdir('/content/wlasl_weights/'))
```

---

## Cell 2: Download WLASL Class Labels

```python
import json
import requests

print("📥 Downloading WLASL class labels...")

# Get WLASL JSON file
!wget -O /content/wlasl_weights/WLASL_v0.3.json https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json

# Load and extract glosses
with open('/content/wlasl_weights/WLASL_v0.3.json', 'r') as f:
    wlasl_data = json.load(f)

# Extract unique glosses
glosses = [item['gloss'] for item in wlasl_data]
print(f"✅ Loaded {len(glosses)} ASL glosses")

# Create mapping
gloss_to_id = {gloss: idx for idx, gloss in enumerate(glosses)}
id_to_gloss = {idx: gloss for idx, gloss in enumerate(glosses)}

# Save mappings
with open('/content/wlasl_weights/gloss_mapping.json', 'w') as f:
    json.dump({'gloss_to_id': gloss_to_id, 'id_to_gloss': id_to_gloss}, f)

print("✅ Class mappings saved!")
print("Sample glosses:", glosses[:10])
```

---

## Cell 3: Clone WLASL Repo for Model Code

```python
import os

print("📥 Cloning WLASL repository...")

if os.path.exists('/content/WLASL'):
    !rm -rf /content/WLASL
    
!git clone https://github.com/dxli94/WLASL.git /content/WLASL

print("✅ Repository cloned!")
```

---

## Cell 4: Test Model Loading

```python
import sys
sys.path.append('/content/WLASL/code/TGCN')

import torch
import numpy as np

# Check structure
print("📂 Model files:")
for root, dirs, files in os.walk('/content/wlasl_weights'):
    for f in files:
        print(f"  {os.path.join(root, f)}")

# The actual model loading will depend on the checkpoint structure
# Let's examine it
import os
archived_path = '/content/wlasl_weights/archived' if os.path.exists('/content/wlasl_weights/archived') else '/content/wlasl_weights'
print(f"\n📂 Checking: {archived_path}")
if os.path.exists(archived_path):
    print(os.listdir(archived_path))
```

---

## Cell 5: Create Simple Inference Wrapper

```python
# This will be customized after we inspect the model structure

class ASLPredictor:
    def __init__(self, model_path, gloss_mapping_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load gloss mapping
        with open(gloss_mapping_path, 'r') as f:
            mapping = json.load(f)
        self.id_to_gloss = {int(k): v for k, v in mapping['id_to_gloss'].items()}
        
        # Load model (structure TBD after inspection)
        print(f"Loading model on {self.device}...")
        # self.model = load_model(model_path)
        
    def predict(self, keypoints):
        """
        keypoints: MediaPipe landmarks flattened
        Returns: (gloss, confidence)
        """
        # Convert MediaPipe format to WLASL format
        # Run inference
        # Return top prediction
        pass

print("✅ ASLPredictor class defined!")
```

---

## Next Steps After Running These Cells:

1. Inspect model checkpoint structure
2. Adapt `predict()` function for MediaPipe input
3. Integrate with our backend `inference_server.py`
4. Test with live camera
