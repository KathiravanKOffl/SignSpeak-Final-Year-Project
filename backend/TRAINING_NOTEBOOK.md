# SignSpeak Training Notebook - Final Version

**Goal:** Train a high-accuracy ASL word recognizer using MediaPipe features.
**Dataset:** WLASL (Word-Level American Sign Language) - Pre-processed MediaPipe features.

---

## 🟢 Cell 1: Setup & Authentication
Run this cell first. It will ask you to upload `kaggle.json`.

```python
# 1. Install libraries
!pip install -q kagglehub torch numpy pandas scikit-learn tqdm

# 2. Upload kaggle.json
from google.colab import files
import os

print("📤 Please upload your kaggle.json file...")
uploaded = files.upload()

# 3. Setup credentials
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

import torch
print(f"✅ Setup complete!")
print(f"🖥️ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## 🟢 Cell 2: Download Dataset
This downloads the pre-processed features (no heavy video downloading needed).

```python
import kagglehub
import os

print("📥 Downloading WLASL MediaPipe features...")
# This uses the credentials you just uploaded
path = kagglehub.dataset_download("risangbaskoro/wlasl-mediapipe-features")

print(f"✅ Downloaded to: {path}")
print("📂 Dataset contents:")
for root, dirs, files in os.walk(path):
    for f in files[:5]:
        print(f"  {f}")
```

---

## 🟢 Cell 3: Load & Prepare Data
We load the top 100 most common signs for a robust demo.

```python
import pandas as pd
import numpy as np
import glob

# 1. Load Parquet files
print("⏳ Loading data files...")
parquet_files = glob.glob(f"{path}/*.parquet")
dfs = []

# Load all files
for f in parquet_files:
    try:
        df = pd.read_parquet(f)
        dfs.append(df)
    except:
        pass

data = pd.concat(dfs, ignore_index=True)
print(f"📊 Total samples loaded: {len(data)}")

# 2. Filter for Top 100 Words
# Column names might vary, finding the label column
label_col = 'label' if 'label' in data.columns else 'sign'
print(f"🏷️ Label column: {label_col}")

# Get top 100 classes
TOP_N = 100
top_classes = data[label_col].value_counts().head(TOP_N).index.tolist()
filtered_data = data[data[label_col].isin(top_classes)].copy()

print(f"✅ Filtered to top {TOP_N} classes ({len(filtered_data)} samples)")
print(f"Sample classes: {top_classes[:10]}...")

# 3. Prepare Features (X) and Labels (y)
# Drop non-feature columns
feature_cols = [c for c in filtered_data.columns if c != label_col and filtered_data[c].dtype in ['float32', 'float64']]
X = filtered_data[feature_cols].fillna(0).values.astype(np.float32)

# Encode labels
unique_labels = sorted(filtered_data[label_col].unique())
label_to_id = {l: i for i, l in enumerate(unique_labels)}
id_to_label = {i: l for i, l in enumerate(unique_labels)}
y = np.array([label_to_id[l] for l in filtered_data[label_col]])

print(f"📐 Input shape: {X.shape}")
print(f"🎯 Output classes: {len(unique_labels)}")

# 4. Save mapping for backend
import json
with open('label_mapping.json', 'w') as f:
    json.dump({'id_to_label': {str(k): v for k, v in id_to_label.items()}}, f)
print("💾 label_mapping.json saved!")
```

---

## 🟢 Cell 4: Create Model
A standardized MLP model for landmark classification.

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class SignDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class SignModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# Split Data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
train_loader = DataLoader(SignDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(SignDataset(X_val, y_val), batch_size=64)

model = SignModel(X.shape[1], len(unique_labels))
print("🤖 Model created!")
```

---

## 🟢 Cell 5: Train (Approx 5-10 mins)
Fast training since we are using pre-extracted features.

```python
import torch.optim as optim
from tqdm.notebook import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

best_acc = 0
EPOCHS = 30

print(f"🚀 Starting training on {device}...")

for epoch in range(EPOCHS):
    model.train()
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
    
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            out = model(X_b)
            correct += (out.argmax(1) == y_b).sum().item()
            total += len(y_b)
    
    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Val Accuracy: {acc:.2f}%")
    
    scheduler.step(acc)
    
    if acc > best_acc:
        best_acc = acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': X.shape[1],
            'num_classes': len(unique_labels),
        }, 'best_model.pth')

print(f"\n🏆 Best Accuracy: {best_acc:.2f}%")
print("💾 Model saved as 'best_model.pth'")
```

---

## 🟢 Cell 6: Deploy to Backend
This updates your backend with the new model.

```python
import shutil
import os

# 1. Update repo code (ensure clean state)
if os.path.exists('/content/SignSpeak-Final-Year-Project'):
    !rm -rf /content/SignSpeak-Final-Year-Project
    
!git clone https://github.com/KathiravanKOffl/SignSpeak-Final-Year-Project.git

# 2. Copy model & mapping
dest = '/content/SignSpeak-Final-Year-Project/backend/checkpoints/'
os.makedirs(dest, exist_ok=True)

shutil.copy('best_model.pth', dest)
shutil.copy('label_mapping.json', dest)

print(f"✅ Model deployed to: {dest}")
print("📂 Files:", os.listdir(dest))

# 3. Start Server
print("\n🚀 RESTARTING SERVER...")
%cd /content/SignSpeak-Final-Year-Project/backend
!python -m uvicorn api.inference_server_wlasl:app --host 0.0.0.0 --port 8000
```
