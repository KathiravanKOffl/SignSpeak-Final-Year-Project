# SignSpeak - WLASL Training (ROBUST VERSION)
# Fixed to prevent browser crashes and widget errors

"""
CHANGES FOR STABILITY:
1. Disabled tqdm widgets (uses simple print statements)
2. Disabled multiprocessing (num_workers=0) - prevents AssertionErrors
3. added 'pin_memory=False' for stability
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 SignSpeak - WLASL ROBUST TRAINING")
print("=" * 80)

# =============================================================================
# Configuration
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✅ Device: {device}")

CONFIG = {
    # Data
    "max_frames": 30,
    "num_landmarks": 75,
    "num_coords": 2,
    
    # Model
    "hidden_dim": 512,
    "num_heads": 8,
    "num_layers": 6,
    "dropout": 0.2,
    
    # Training
    "batch_size": 128,
    "epochs": 150,
    "learning_rate": 3e-4,
    "weight_decay": 0.001,
    "label_smoothing": 0.1,
    
    # Stability
    "num_workers": 0,    # FIX: Disable multiprocessing
    "pin_memory": False, # FIX: Disable pinning
}

# =============================================================================
# Load Data
# =============================================================================

print("\n" + "=" * 80)
print("📥 LOADING WLASL DATASET")
print("=" * 80)

data_path = Path("/kaggle/input/land-mark-holistic-featuresfor-wlasl")
# Fallback search if path is different
if not data_path.exists():
    print("⚠️ Path not found, searching...")
    for p in Path("/kaggle/input").rglob("x_output*.npy"):
        data_path = p.parent
        print(f"Found at: {data_path}")
        break

x_file = list(data_path.glob("x_output*.npy"))[0]
y_file = list(data_path.glob("y_output*.npy"))[0]

print(f"1️⃣ Loading features from: {x_file.name}")
X_raw = np.load(x_file)
print(f"   Shape: {X_raw.shape}")

print(f"2️⃣ Loading labels from: {y_file.name}")
y_raw = np.load(y_file)

# Decode labels
if y_raw.dtype.kind == 'S':
    y_labels = np.array([label.decode('utf-8') if isinstance(label, bytes) else str(label) 
                         for label in y_raw])
else:
    y_labels = y_raw.astype(str)

# Encode
unique_labels = sorted(set(y_labels))
label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
id_to_label = {idx: label for idx, label in enumerate(unique_labels)}
y = np.array([label_to_id[label] for label in y_labels])

num_classes = len(unique_labels)
print(f"   Classes: {num_classes}")

# Clean data
X_raw = np.nan_to_num(X_raw, 0).astype(np.float32)

# =============================================================================
# Augmentation
# =============================================================================

class TemporalAugmentation:
    @staticmethod
    def rotate(X, angle_range=10):
        angle = np.random.uniform(-angle_range, angle_range) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = X.copy()
        for t in range(X.shape[0]):
            rotated[t] = X[t] @ rotation_matrix.T
        return rotated
    
    @staticmethod
    def scale(X, scale_range=0.1):
        scale = np.random.uniform(1 - scale_range, 1 + scale_range)
        return X * scale
    
    @staticmethod
    def apply(X):
        if np.random.random() < 0.5:
            X = TemporalAugmentation.rotate(X)
        if np.random.random() < 0.5:
            X = TemporalAugmentation.scale(X)
        return X

# =============================================================================
# Dataset
# =============================================================================

class WLASLDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment:
            x = TemporalAugmentation.apply(x)
        # Flatten: (30, 75, 2) -> (30, 150)
        x = x.reshape(x.shape[0], -1)
        return torch.FloatTensor(x), torch.LongTensor([self.y[idx]])[0]

X_train, X_val, y_train, y_val = train_test_split(
    X_raw, y, test_size=0.15, random_state=42, stratify=y
)

train_loader = DataLoader(
    WLASLDataset(X_train, y_train, augment=True), 
    batch_size=CONFIG["batch_size"], 
    shuffle=True, 
    num_workers=CONFIG["num_workers"], # 0
    pin_memory=CONFIG["pin_memory"]    # False
)

val_loader = DataLoader(
    WLASLDataset(X_val, y_val, augment=False), 
    batch_size=CONFIG["batch_size"], 
    num_workers=CONFIG["num_workers"], # 0
    pin_memory=CONFIG["pin_memory"]    # False
)

# =============================================================================
# Model
# =============================================================================

class SpatialTemporalTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, num_classes, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 30, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_classes)
        )
    
    def forward(self, x):
        x = self.input_proj(x) + self.pos_encoding
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.temporal_pool(x).squeeze(-1)
        return self.classifier(x)

input_dim = 150
model = SpatialTemporalTransformer(
    input_dim=input_dim, hidden_dim=CONFIG["hidden_dim"],
    num_heads=CONFIG["num_heads"], num_layers=CONFIG["num_layers"],
    num_classes=num_classes, dropout=CONFIG["dropout"]
).to(device)

# =============================================================================
# Training Loop
# =============================================================================

criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=CONFIG["learning_rate"], total_steps=len(train_loader)*CONFIG["epochs"],
    pct_start=0.1, anneal_strategy='cos'
)

print("\n" + "=" * 80)
print("🚀 TRAINING STARTED (Simple Mode)")
print("=" * 80)

best_val_acc = 0
start_time = time.time()

for epoch in range(CONFIG["epochs"]):
    # Train
    model.train()
    train_loss, correct, total = 0, 0, 0
    
    for i, (X_b, y_b) in enumerate(train_loader):
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        out = model(X_b)
        loss = criterion(out, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        correct += (out.argmax(1) == y_b).sum().item()
        total += y_b.size(0)
    
    train_acc = 100. * correct / total
    avg_train_loss = train_loss / len(train_loader)
    
    # Val
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            out = model(X_b)
            loss = criterion(out, y_b)
            val_loss += loss.item()
            correct += (out.argmax(1) == y_b).sum().item()
            total += y_b.size(0)
            
    val_acc = 100. * correct / total
    avg_val_loss = val_loss / len(val_loader)
    
    elapsed = (time.time() - start_time) / 60
    
    print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
          f"Train: {avg_train_loss:.4f} ({train_acc:.2f}%) | "
          f"Val: {avg_val_loss:.4f} ({val_acc:.2f}%) | "
          f"Time: {elapsed:.1f}m")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print(f"  💾 Saved New Best: {best_val_acc:.2f}%")
        torch.save({
            'model': model.state_dict(),
            'config': CONFIG,
            'id_to_label': id_to_label,
            'input_dim': input_dim,
            'num_classes': num_classes
        }, "best_model.pth")

# Save labels
with open("label_mapping.json", "w") as f:
    json.dump({"id_to_label": {str(k):v for k,v in id_to_label.items()}}, f)

print(f"\n✅ DONE! Best Acc: {best_val_acc:.2f}%")
