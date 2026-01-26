# SignSpeak Model Training - MediaPipe Landmarks

Training notebook for ASL recognition on MediaPipe landmark format.
Uses Google's Kaggle ASL competition dataset (same 543-landmark format as our app).

---

## Cell 1: Setup Kaggle API

```python
# Upload your kaggle.json first!
# Get it from: https://www.kaggle.com/settings → API → Create New Token

import os
os.makedirs('/root/.kaggle', exist_ok=True)

# If you uploaded kaggle.json to Colab:
!cp /content/kaggle.json /root/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json

!pip install -q kaggle
print("✅ Kaggle API ready!")
```

---

## Cell 2: Download ASL Dataset

```python
# Google ASL Fingerspelling Dataset - 543 landmarks per frame
!kaggle competitions download -c asl-fingerspelling

# Or use smaller processed dataset:
# !kaggle datasets download -d alancarlosgomes/mediapipe-processed-asl-dataset

!unzip -q asl-fingerspelling.zip -d /content/asl_data
print("✅ Dataset downloaded!")
```

---

## Cell 3: Alternative - Use Pre-Processed Landmarks

```python
# Smaller dataset for quick training
!kaggle datasets download -d alancarlosgomes/mediapipe-processed-asl-dataset
!unzip -q mediapipe-processed-asl-dataset.zip -d /content/asl_landmarks

import os
print("📂 Contents:", os.listdir('/content/asl_landmarks'))
```

---

## Cell 4: Load and Prepare Data

```python
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import json

# Load landmark data
data_path = '/content/asl_landmarks'  # Adjust based on dataset

# Check what files we have
for root, dirs, files in os.walk(data_path):
    for f in files[:10]:
        print(os.path.join(root, f))
```

---

## Cell 5: Create Dataset Class

```python
class ASLLandmarkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Assuming landmarks are in shape (N, 225) for 75 keypoints * 3 coords
# Or (N, 543) for full MediaPipe holistic

print("✅ Dataset class ready!")
```

---

## Cell 6: Define Model

```python
import torch.nn as nn

class ASLClassifier(nn.Module):
    def __init__(self, input_dim=225, hidden_dim=512, num_classes=100):
        super().__init__()
        
        self.encoder = nn.Sequential(
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
        )
        
        self.classifier = nn.Linear(hidden_dim // 4, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len, features) or (batch, features)
        if len(x.shape) == 3:
            x = x.mean(dim=1)  # Temporal average pooling
        
        features = self.encoder(x)
        return self.classifier(features)

# Test model
model = ASLClassifier(input_dim=225, num_classes=100)
test_input = torch.randn(4, 225)
output = model(test_input)
print(f"✅ Model output shape: {output.shape}")  # Should be (4, 100)
```

---

## Cell 7: Training Loop

```python
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, val_loader, epochs=50, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for X, y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        
        acc = 100 * correct / total
        history['val_acc'].append(acc)
        
        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={acc:.2f}%')
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': acc,
            }, '/content/best_asl_model.pth')
            print(f'  ✅ Saved best model (acc={acc:.2f}%)')
        
        scheduler.step()
    
    return history

print("✅ Training function ready!")
```

---

## Cell 8: Quick Demo Training (Synthetic Data)

```python
# Quick test with synthetic data to verify pipeline works
print("🧪 Testing with synthetic data...")

# Create dummy data
num_samples = 1000
num_classes = 26  # A-Z alphabet
input_dim = 225  # 75 keypoints * 3 coords

X = np.random.randn(num_samples, input_dim).astype(np.float32)
y = np.random.randint(0, num_classes, num_samples)

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Create loaders
train_dataset = ASLLandmarkDataset(X_train, y_train)
val_dataset = ASLLandmarkDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Quick train (5 epochs just to test)
model = ASLClassifier(input_dim=input_dim, num_classes=num_classes)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Train 5 epochs
history = train_model(model, train_loader, val_loader, epochs=5, device=device)
print("\n✅ Pipeline works! Now load real data.")
```

---

## Cell 9: Copy Trained Model to Backend

```python
import shutil

# Copy model to repo
model_path = '/content/best_asl_model.pth'
dest_path = '/content/SignSpeak-Final-Year-Project/backend/checkpoints/'

os.makedirs(dest_path, exist_ok=True)
shutil.copy(model_path, dest_path + 'asl_landmark_model.pth')

print(f"✅ Model copied to {dest_path}")
```

---

## Cell 10: Update Inference Server

After training, update the inference server to load your trained model:

```python
# Modify inference_server_wlasl.py to load:
# checkpoint_path = '/content/SignSpeak-Final-Year-Project/backend/checkpoints/asl_landmark_model.pth'
```

---

## Next Steps

1. **Upload kaggle.json** to Colab
2. **Run Cell 2 or 3** to download real dataset
3. **Modify Cell 4** to load your downloaded data
4. **Run Cell 7** with real data for full training
5. **Run Cell 9** to copy model
6. **Restart server** with trained model
