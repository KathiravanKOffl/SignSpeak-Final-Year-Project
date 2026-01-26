# SignSpeak Model Training - MediaPipe Landmarks

Training ASL Alphabet (A-Z) recognition using MediaPipe landmarks.

---

## Cell 1: Install Dependencies

```python
!pip install -q kagglehub torch numpy pandas scikit-learn tqdm

import torch
print(f"✅ PyTorch {torch.__version__}")
print(f"🖥️ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
```

---

## Cell 2: Download Dataset (kagglehub)

```python
import kagglehub

# Download MediaPipe-processed ASL dataset
path = kagglehub.dataset_download("risangbaskoro/asl-words-wlasl-mediapipe-features")
print(f"✅ Downloaded to: {path}")

# List contents
import os
for root, dirs, files in os.walk(path):
    for f in files[:10]:
        print(os.path.join(root, f))
```

---

## Cell 3: Load Data

```python
import numpy as np
import pandas as pd
import os

# Find parquet/csv files
data_path = path
files = []
for root, dirs, fs in os.walk(data_path):
    for f in fs:
        if f.endswith('.parquet') or f.endswith('.csv') or f.endswith('.npy'):
            files.append(os.path.join(root, f))
            
print(f"Found files: {files[:5]}")

# Load based on file type
if files[0].endswith('.parquet'):
    df = pd.read_parquet(files[0])
elif files[0].endswith('.csv'):
    df = pd.read_csv(files[0])
    
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)[:10]}")
```

---

## Cell 4: Prepare Data

```python
# Adjust these based on actual dataset structure
# Common formats:
# - 'landmark_0_x', 'landmark_0_y', 'landmark_0_z', ... (flattened)
# - Separate columns for each landmark coordinate

# Get feature columns (exclude label column)
label_col = 'label' if 'label' in df.columns else 'sign' if 'sign' in df.columns else df.columns[-1]
feature_cols = [c for c in df.columns if c != label_col]

print(f"Label column: {label_col}")
print(f"Feature columns: {len(feature_cols)}")
print(f"Unique labels: {df[label_col].nunique()}")

# Prepare X and y
X = df[feature_cols].values.astype(np.float32)
y_labels = df[label_col].values

# Create label mapping
unique_labels = sorted(set(y_labels))
label_to_id = {label: i for i, label in enumerate(unique_labels)}
id_to_label = {i: label for label, i in label_to_id.items()}

y = np.array([label_to_id[label] for label in y_labels])

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Classes: {len(unique_labels)}")
print(f"Sample labels: {unique_labels[:10]}")

# Save mapping
import json
with open('/content/label_mapping.json', 'w') as f:
    json.dump({'label_to_id': label_to_id, 'id_to_label': {str(k): v for k, v in id_to_label.items()}}, f)
print("✅ Label mapping saved!")
```

---

## Cell 5: Create Dataset and Model

```python
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn

class ASLDataset(Dataset):
    def __init__(self, X, y):
        # Handle NaN values
        X = np.nan_to_num(X, 0)
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ASLClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.mean(dim=1)
        return self.net(x)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_dataset = ASLDataset(X_train, y_train)
val_dataset = ASLDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Create model
input_dim = X.shape[1]
num_classes = len(unique_labels)
model = ASLClassifier(input_dim=input_dim, num_classes=num_classes)

print(f"✅ Model ready: input_dim={input_dim}, num_classes={num_classes}")
```

---

## Cell 6: Train Model

```python
import torch.optim as optim
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

best_acc = 0
EPOCHS = 50

for epoch in range(EPOCHS):
    # Train
    model.train()
    total_loss = 0
    for X_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}', leave=False):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
    
    acc = 100 * correct / total
    print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Val Acc={acc:.2f}%')
    
    if acc > best_acc:
        best_acc = acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': input_dim,
            'num_classes': num_classes,
            'accuracy': acc,
            'id_to_label': id_to_label,
        }, '/content/best_asl_model.pth')
        print(f'  ✅ Best model saved! ({acc:.2f}%)')
    
    scheduler.step()

print(f'\n🎉 Training complete! Best accuracy: {best_acc:.2f}%')
```

---

## Cell 7: Copy to Backend

```python
import shutil
import os

# Clone/update repo
if not os.path.exists('/content/SignSpeak-Final-Year-Project'):
    !git clone https://github.com/KathiravanKOffl/SignSpeak-Final-Year-Project.git
else:
    %cd /content/SignSpeak-Final-Year-Project
    !git pull

# Copy model and mapping
dest = '/content/SignSpeak-Final-Year-Project/backend/checkpoints/'
os.makedirs(dest, exist_ok=True)
shutil.copy('/content/best_asl_model.pth', dest)
shutil.copy('/content/label_mapping.json', dest)

print(f"✅ Model copied to {dest}")
print("📂 Files:", os.listdir(dest))
```

---

## Cell 8: Test Locally

```python
# Quick test
checkpoint = torch.load('/content/best_asl_model.pth')
print(f"Model accuracy: {checkpoint['accuracy']:.2f}%")
print(f"Classes: {checkpoint['num_classes']}")
print(f"Sample labels: {list(checkpoint['id_to_label'].values())[:10]}")
```

---

## After Training:

1. Update `inference_server_wlasl.py` to load `/checkpoints/best_asl_model.pth`
2. Restart server
3. Test with camera!
