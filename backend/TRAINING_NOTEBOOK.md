# SignSpeak Model Training - Complete Guide

Train ASL word-level sign recognition on MediaPipe landmarks.

---

## Cell 1: Upload Kaggle Token & Install

```python
# Upload kaggle.json (drag and drop when prompted)
from google.colab import files
print("📤 Upload your kaggle.json file...")
uploaded = files.upload()

# Setup Kaggle
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Install dependencies
!pip install -q kagglehub torch numpy pandas scikit-learn tqdm

import torch
print(f"✅ Setup complete! GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## Cell 2: Download Dataset

```python
import kagglehub
import os

# Download WLASL MediaPipe features (word-level signs)
print("📥 Downloading dataset...")
path = kagglehub.dataset_download("risangbaskoro/wlasl-mediapipe-features")
print(f"✅ Downloaded to: {path}")

# Show contents
for root, dirs, files_list in os.walk(path):
    for f in files_list[:10]:
        print(f"  {os.path.join(root, f)}")
```

---

## Cell 3: Load Data

```python
import pandas as pd
import numpy as np
import os

# Find parquet files
data_files = []
for root, dirs, files_list in os.walk(path):
    for f in files_list:
        if f.endswith('.parquet'):
            data_files.append(os.path.join(root, f))
            
print(f"Found {len(data_files)} parquet files")

# Load all data
dfs = []
for f in data_files[:50]:  # Limit for faster training
    try:
        df = pd.read_parquet(f)
        dfs.append(df)
    except Exception as e:
        print(f"Skipping {f}: {e}")

data = pd.concat(dfs, ignore_index=True)
print(f"✅ Loaded {len(data)} samples")
print(f"Columns: {list(data.columns)[:10]}")
```

---

## Cell 4: Prepare Features

```python
# Find label and feature columns
if 'label' in data.columns:
    label_col = 'label'
elif 'sign' in data.columns:
    label_col = 'sign'
else:
    label_col = data.columns[-1]
    
feature_cols = [c for c in data.columns if c != label_col and data[c].dtype in ['float64', 'float32', 'int64']]

print(f"Label column: {label_col}")
print(f"Features: {len(feature_cols)}")

# Remove NaN rows
data = data.dropna(subset=[label_col])

# Get top N most common classes (for faster training)
N_CLASSES = 50
top_classes = data[label_col].value_counts().head(N_CLASSES).index.tolist()
data = data[data[label_col].isin(top_classes)]

print(f"Using top {N_CLASSES} classes: {top_classes[:10]}...")

# Prepare X, y
X = data[feature_cols].fillna(0).values.astype(np.float32)
y_labels = data[label_col].values

# Create mappings
unique_labels = sorted(set(y_labels))
label_to_id = {label: i for i, label in enumerate(unique_labels)}
id_to_label = {i: label for label, i in label_to_id.items()}
y = np.array([label_to_id[l] for l in y_labels])

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Classes: {len(unique_labels)}")

# Save mapping
import json
with open('/content/label_mapping.json', 'w') as f:
    json.dump({'id_to_label': {str(k): v for k, v in id_to_label.items()}}, f)
print("✅ Saved label_mapping.json")
```

---

## Cell 5: Create Model

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class ASLDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(np.nan_to_num(X, 0))
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class ASLModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        if len(x.shape) == 3: x = x.mean(1)
        return self.net(x)

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
train_loader = DataLoader(ASLDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(ASLDataset(X_val, y_val), batch_size=64)

model = ASLModel(X.shape[1], len(unique_labels))
print(f"✅ Model: {X.shape[1]} features → {len(unique_labels)} classes")
```

---

## Cell 6: Train (30-60 min)

```python
import torch.optim as optim
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

best_acc = 0
for epoch in range(50):
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
            correct += (model(X_b).argmax(1) == y_b).sum().item()
            total += len(y_b)
    
    acc = 100 * correct / total
    print(f'Epoch {epoch+1}: {acc:.1f}%')
    
    if acc > best_acc:
        best_acc = acc
        torch.save({'model': model.state_dict(), 'input_dim': X.shape[1], 
                    'num_classes': len(unique_labels), 'id_to_label': id_to_label}, 
                   '/content/best_model.pth')
        print(f'  ✅ Saved! Best: {acc:.1f}%')

print(f'\n🎉 Done! Best accuracy: {best_acc:.1f}%')
```

---

## Cell 7: Copy to Backend

```python
import shutil, os

# Update repo
!cd /content && rm -rf SignSpeak* && git clone https://github.com/KathiravanKOffl/SignSpeak-Final-Year-Project.git

# Copy model
dest = '/content/SignSpeak-Final-Year-Project/backend/checkpoints/'
os.makedirs(dest, exist_ok=True)
shutil.copy('/content/best_model.pth', dest)
shutil.copy('/content/label_mapping.json', dest)
print(f"✅ Copied to {dest}")
```

---

## Cell 8: Restart Server

```python
%cd /content/SignSpeak-Final-Year-Project/backend
!python -m uvicorn api.inference_server_wlasl:app --host 0.0.0.0 --port 8000 --reload
```
