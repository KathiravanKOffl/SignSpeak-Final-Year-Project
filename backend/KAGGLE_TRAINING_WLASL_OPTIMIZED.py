# SignSpeak - WLASL Training (OPTIMIZED FOR >90% ACCURACY)
# Based on research: VideoMAE + Spatial-Temporal Transformer

"""
GOAL: Achieve >90% validation accuracy on WLASL

IMPROVEMENTS OVER BASIC VERSION:
1. ✅ Full temporal modeling (use all 30 frames, not averaged)
2. ✅ Spatial-Temporal Transformer architecture
3. ✅ Advanced data augmentation
4. ✅ Warmup + Cosine schedule
5. ✅ Mixed precision training (faster)
6. ✅ Better regularization

SETUP:
1. Kaggle → New Notebook
2. Add: "land-mark-holistic-features-WLASL"
3. GPU P100 or T4
4. Copy entire script → Run All
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🚀 SignSpeak - WLASL OPTIMIZED (Target: >90% Accuracy)")
print("=" * 80)

# =============================================================================
# Configuration
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✅ Device: {device}")

CONFIG = {
    # Data
    "max_frames": 30,       # Use full temporal sequence
    "num_landmarks": 75,    # Hand + pose landmarks
    "num_coords": 2,        # x, y coordinates
    
    # Model
    "hidden_dim": 512,
    "num_heads": 8,
    "num_layers": 6,
    "dropout": 0.2,         # Reduced dropout (data is clean)
    
    # Training
    "batch_size": 128,      # Larger batch for stability
    "epochs": 150,
    "learning_rate": 3e-4,
    "weight_decay": 0.001,
    "label_smoothing": 0.1,
    
    # Augmentation
    "augment": True,
    "aug_rotation": 10,     # degrees
    "aug_scale": 0.1,       # 10% scale variation
    "aug_temporal_mask": 0.1,  # Mask 10% of frames
}

# =============================================================================
# Load Data
# =============================================================================

print("\n" + "=" * 80)
print("📥 LOADING WLASL DATASET")
print("=" * 80)

data_path = Path("/kaggle/input/land-mark-holistic-featuresfor-wlasl")
X_raw = np.load(data_path / "x_output (1).npy")
y_raw = np.load(data_path / "y_output.npy")

print(f"\n✅ Loaded data:")
print(f"   Features: {X_raw.shape}")  # (11980, 30, 75, 2)
print(f"   Labels: {y_raw.shape}")

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
print(f"\n✅ Classes: {num_classes}")
print(f"   Sample labels: {unique_labels[:10]}")

# Clean data
X_raw = np.nan_to_num(X_raw, 0).astype(np.float32)

# =============================================================================
# Data Augmentation
# =============================================================================

class TemporalAugmentation:
    """Data augmentation for sign language sequences"""
    
    @staticmethod
    def rotate(X, angle_range=10):
        """Rotate landmarks by random angle"""
        angle = np.random.uniform(-angle_range, angle_range) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # Apply rotation to each frame
        rotated = X.copy()
        for t in range(X.shape[0]):
            rotated[t] = X[t] @ rotation_matrix.T
        return rotated
    
    @staticmethod
    def scale(X, scale_range=0.1):
        """Random scaling"""
        scale = np.random.uniform(1 - scale_range, 1 + scale_range)
        return X * scale
    
    @staticmethod
    def temporal_mask(X, mask_prob=0.1):
        """Randomly mask some frames"""
        mask = np.random.random(X.shape[0]) > mask_prob
        X_masked = X.copy()
        X_masked[~mask] = 0
        return X_masked
    
    @staticmethod
    def apply(X, config):
        """Apply random augmentations"""
        if np.random.random() < 0.5:
            X = TemporalAugmentation.rotate(X, config["aug_rotation"])
        if np.random.random() < 0.5:
            X = TemporalAugmentation.scale(X, config["aug_scale"])
        if np.random.random() < 0.3:
            X = TemporalAugmentation.temporal_mask(X, config["aug_temporal_mask"])
        return X

# =============================================================================
# Dataset
# =============================================================================

class WLASLDataset(Dataset):
    def __init__(self, X, y, config, augment=False):
        self.X = X  # (N, 30, 75, 2)
        self.y = y
        self.config = config
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].copy()  # (30, 75, 2)
        
        # Apply augmentation
        if self.augment:
            x = TemporalAugmentation.apply(x, self.config)
        
        # Flatten spatial dims: (30, 75, 2) -> (30, 150)
        x = x.reshape(x.shape[0], -1)
        
        return torch.FloatTensor(x), torch.LongTensor([self.y[idx]])[0]

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X_raw, y, test_size=0.15, random_state=42, stratify=y
)

train_dataset = WLASLDataset(X_train, y_train, CONFIG, augment=True)
val_dataset = WLASLDataset(X_val, y_val, CONFIG, augment=False)

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], 
                          shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], 
                        num_workers=2, pin_memory=True)

print(f"\n✅ Data split:")
print(f"   Train: {len(X_train):,} | Val: {len(X_val):,}")

# =============================================================================
# Model: Spatial-Temporal Transformer
# =============================================================================

class SpatialTemporalTransformer(nn.Module):
    """
    Advanced architecture for sign language recognition
    - Processes full temporal sequences
    - Multi-head attention over time
    - Positional encoding for temporal info
    """
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, num_classes, dropout=0.2):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding for temporal information
        self.pos_encoding = nn.Parameter(torch.randn(1, 30, hidden_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, time=30, features=150)
        batch_size = x.size(0)
        
        # Project input
        x = self.input_proj(x)  # (batch, 30, hidden)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Transformer
        x = self.transformer(x)  # (batch, 30, hidden)
        
        # Temporal pooling: (batch, 30, hidden) -> (batch, hidden)
        x = x.transpose(1, 2)  # (batch, hidden, 30)
        x = self.temporal_pool(x).squeeze(-1)  # (batch, hidden)
        
        # Classify
        return self.classifier(x)

# Initialize
input_dim = CONFIG["num_landmarks"] * CONFIG["num_coords"]  # 150
model = SpatialTemporalTransformer(
    input_dim=input_dim,
    hidden_dim=CONFIG["hidden_dim"],
    num_heads=CONFIG["num_heads"],
    num_layers=CONFIG["num_layers"],
    num_classes=num_classes,
    dropout=CONFIG["dropout"]
).to(device)

print(f"\n✅ Model: {sum(p.numel() for p in model.parameters()):,} parameters")

# =============================================================================
# Training
# =============================================================================

criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], 
                        weight_decay=CONFIG["weight_decay"])

# Warmup + Cosine annealing
total_steps = len(train_loader) * CONFIG["epochs"]
warmup_steps = len(train_loader) * 10

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=CONFIG["learning_rate"],
    total_steps=total_steps,
    pct_start=0.1,
    anneal_strategy='cos'
)

# Mixed precision training (faster on modern GPUs)
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

def train_epoch(model, loader):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for X_batch, y_batch in pbar:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(y_batch).sum().item()
        total += y_batch.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += y_batch.size(0)
    
    return total_loss / len(loader), 100. * correct / total

# Training loop
print("\n" + "=" * 80)
print("🚀 TRAINING (Target: >90% Val Acc)")
print("=" * 80)

best_val_acc = 0
patience_counter = 0
patience = 20

for epoch in range(CONFIG["epochs"]):
    train_loss, train_acc = train_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)
    
    print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
          f"Train: {train_loss:.4f} ({train_acc:.2f}%) | "
          f"Val: {val_loss:.4f} ({val_acc:.2f}%)")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'config': CONFIG,
            'id_to_label': id_to_label,
            'input_dim': input_dim,
            'num_classes': num_classes,
        }, "best_model.pth")
        
        print(f"  💾 Saved! (Best: {best_val_acc:.2f}%)")
        
        if best_val_acc >= 90.0:
            print(f"\n🎯 TARGET ACHIEVED! Val Acc: {best_val_acc:.2f}% ≥ 90%")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping")
            break

print(f"\n🏆 FINAL: {best_val_acc:.2f}% Validation Accuracy")

# Save label mapping
with open("label_mapping.json", "w") as f:
    json.dump({
        "id_to_label": {str(k): v for k, v in id_to_label.items()},
        "label_to_id": {v: k for k, v in label_to_id.items()},
        "num_classes": num_classes,
        "input_dim": input_dim,
        "best_val_acc": float(best_val_acc),
        "language": "ASL",
        "dataset": "WLASL"
    }, f, indent=2)

print("\n📦 Download: best_model.pth + label_mapping.json")
