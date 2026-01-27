# SignSpeak Training - Synthetic Data (Guaranteed to Work) 🛡️

Since external downloads are blocked/broken, we will train on **Generated Synthetic Data**.
This allows us to test the **Full Pipeline** (Frontend <-> Backend) immediately.

---

## 🟢 Cell 1: Setup

```python
!pip install -q torch numpy pandas scikit-learn

import torch
import numpy as np
import os
import json
import shutil
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

print(f"✅ Setup complete! GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## 🟢 Cell 2: Generate Synthetic Data
We create fake landmark data for 3 classes: `hello`, `thanks`, `iloveyou`.

```python
print("🎲 Generating synthetic data...")

# Parameters matching our MediaPipe input (225 features)
NUM_SAMPLES = 1000
NUM_CLASSES = 3
INPUT_DIM = 225 
CLASSES = ['hello', 'thanks', 'iloveyou']

# Generate random features
X = np.random.randn(NUM_SAMPLES, INPUT_DIM).astype(np.float32)

# Generate labels (0, 1, 2)
y = np.random.randint(0, NUM_CLASSES, NUM_SAMPLES)

print(f"✅ Generated {NUM_SAMPLES} samples.")
print(f"Classes: {CLASSES}")

# Create mapping
label_mapping = {
    'id_to_label': {str(i): c for i, c in enumerate(CLASSES)},
    'label_to_id': {c: i for i, c in enumerate(CLASSES)}
}

# Save mapping
with open('label_mapping.json', 'w') as f:
    json.dump(label_mapping, f)
print("✅ Saved label_mapping.json")
```

---

## 🟢 Cell 3: Train Model (Fast)

```python
# Dataset Class
class SyntheticDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# Simple Model
model = nn.Sequential(
    nn.Linear(INPUT_DIM, 64),
    nn.ReLU(),
    nn.Linear(64, NUM_CLASSES)
)

# Train Loop
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

train_loader = DataLoader(SyntheticDataset(X, y), batch_size=32, shuffle=True)

print("🚀 Training...")
for epoch in range(5):
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done.")

# Save
torch.save({
    'model_state_dict': model.state_dict(),
    'num_classes': NUM_CLASSES,
    'input_dim': INPUT_DIM
}, 'best_model.pth')

print("✅ Model trained and saved as 'best_model.pth'")
```

---

## 🟢 Cell 4: Deploy & Restart Server

```python
# 1. Clone/Update Repo
if os.path.exists('/content/SignSpeak-Final-Year-Project'):
    !rm -rf /content/SignSpeak-Final-Year-Project
!git clone https://github.com/KathiravanKOffl/SignSpeak-Final-Year-Project.git

# 2. Deploy Model
dest = '/content/SignSpeak-Final-Year-Project/backend/checkpoints/'
os.makedirs(dest, exist_ok=True)
shutil.copy('best_model.pth', dest)
shutil.copy('label_mapping.json', dest)
print(f"✅ Model deployed to {dest}")

# 3. Start Server
print("\n🚀 RESTARTING SERVER...")
%cd /content/SignSpeak-Final-Year-Project/backend
!python -m uvicorn api.inference_server_wlasl:app --host 0.0.0.0 --port 8000
```
