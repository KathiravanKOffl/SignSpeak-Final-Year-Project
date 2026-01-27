# SignSpeak - Final Kaggle Training Script
# Optimized for the actual WLASL dataset structure found on Kaggle

"""
## BASED ON DIAGNOSTIC OUTPUT:
Dataset: land-mark-holistic-featuresfor-wlasl
- x_output (1).npy: Shape (11980, 30, 75, 2) → 11,980 samples, 30 frames, 75 landmarks, x/y coords
- y_output.npy: Shape (11980,) → Labels as byte strings

## SETUP INSTRUCTIONS:
1. Kaggle → New Notebook
2. Add dataset: "land-mark-holistic-features-WLASL"
3. Settings → GPU P100 or T4
4. Copy this ENTIRE script → Run All
5. Close laptop → Training runs 4-8 hours
6. Download: best_model.pth + label_mapping.json
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
print("🚀 SignSpeak - WLASL Training (FINAL VERSION)")
print("=" * 80)

# =============================================================================
# CELL 1: Configuration
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✅ Device: {device}")

CONFIG = {
    "input_dim": None,  # Will be calculated from data
    "hidden_dim": 512,  # Increased for better capacity
    "num_heads": 8,
    "num_layers": 6,    # Increased depth
    "num_classes": None,  # Will be calculated from labels
    "batch_size": 64,   # Larger batch for better gradient estimates
    "epochs": 100,
    "learning_rate": 2e-4,
    "dropout": 0.3,
    "weight_decay": 0.01,
}

# =============================================================================
# CELL 2: Load Data (THE KEY PART!)
# =============================================================================

print("\n" + "=" * 80)
print("📥 LOADING WLASL DATASET")
print("=" * 80)

# Path to dataset
data_path = Path("/kaggle/input/land-mark-holistic-featuresfor-wlasl")

# Load feature file
x_file = data_path / "x_output (1).npy"
y_file = data_path / "y_output.npy"

print(f"\n1️⃣ Loading features from: {x_file.name}")
X_raw = np.load(x_file)
print(f"   ✅ Loaded: {X_raw.shape}")
print(f"   Format: ({X_raw.shape[0]} samples, {X_raw.shape[1]} frames, {X_raw.shape[2]} landmarks, {X_raw.shape[3]} coords)")

print(f"\n2️⃣ Loading labels from: {y_file.name}")
y_raw = np.load(y_file)
print(f"   ✅ Loaded: {y_raw.shape}")

# Decode byte string labels
if y_raw.dtype.kind == 'S':  # Byte strings
    y_labels = np.array([label.decode('utf-8') if isinstance(label, bytes) else str(label) for label in y_raw])
else:
    y_labels = y_raw.astype(str)

print(f"\n3️⃣ Processing data...")

# Reshape features: (samples, frames, landmarks, coords) → (samples, features)
# Strategy: Flatten spatial (landmarks, coords) then average over time
n_samples, n_frames, n_landmarks, n_coords = X_raw.shape

# Option A: Flatten everything and average over frames
# (11980, 30, 75, 2) → (11980, 30, 150) → (11980, 150)
X_reshaped = X_raw.reshape(n_samples, n_frames, n_landmarks * n_coords)
X = X_reshaped.mean(axis=1).astype(np.float32)  # Average over frames
print(f"   Averaged features over time: {X.shape}")

# Option B: Use all frames (for sequence models)
# X_sequential = X_reshaped  # Keep (11980, 30, 150) for LSTM/Transformer if needed

# Encode labels
unique_labels = sorted(set(y_labels))
label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
id_to_label = {idx: label for idx, label in enumerate(unique_labels)}
y = np.array([label_to_id[label] for label in y_labels])

# Update config
CONFIG["num_classes"] = len(unique_labels)
CONFIG["input_dim"] = X.shape[1]

print(f"\n✅ Dataset Summary:")
print(f"   Total samples: {len(X):,}")
print(f"   Features per sample: {CONFIG['input_dim']}")
print(f"   Number of classes: {CONFIG['num_classes']}")
print(f"   Sample labels: {unique_labels[:10]}")

# Verify data quality
print(f"\n4️⃣ Data Quality Checks:")
print(f"   NaN values: {np.isnan(X).sum()}")
print(f"   Inf values: {np.isinf(X).sum()}")
print(f"   Feature range: [{X.min():.3f}, {X.max():.3f}]")

# Clean data
X = np.nan_to_num(X, 0)  # Replace NaN with 0

# =============================================================================
# CELL 3: Dataset & DataLoader
# =============================================================================

class SignDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Stratified split to maintain class balance
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

train_dataset = SignDataset(X_train, y_train)
val_dataset = SignDataset(X_val, y_val)

train_loader = DataLoader(
    train_dataset, 
    batch_size=CONFIG["batch_size"], 
    shuffle=True, 
    num_workers=2,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=CONFIG["batch_size"], 
    num_workers=2,
    pin_memory=True
)

print(f"\n✅ Data Split:")
print(f"   Training: {len(X_train):,} samples")
print(f"   Validation: {len(X_val):,} samples")
print(f"   Batches per epoch: {len(train_loader)}")

# =============================================================================
# CELL 4: Model Architecture (Hybrid CNN-Transformer)
# =============================================================================

class HybridSignTransformer(nn.Module):
    """
    Hybrid architecture combining:
    - Spatial feature extraction (CNN-style)
    - Temporal modeling (Transformer)
    - Classification head
    """
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, num_classes, dropout=0.3):
        super().__init__()
        
        # Spatial Feature Encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Transformer Encoder for temporal patterns
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, features)
        x = self.spatial_encoder(x)         # (batch, hidden_dim)
        x = x.unsqueeze(1)                   # (batch, 1, hidden_dim) - add sequence dim
        x = self.transformer(x)              # (batch, 1, hidden_dim)
        x = x.squeeze(1)                     # (batch, hidden_dim)
        return self.classifier(x)            # (batch, num_classes)

# Initialize model
model = HybridSignTransformer(
    input_dim=CONFIG["input_dim"],
    hidden_dim=CONFIG["hidden_dim"],
    num_heads=CONFIG["num_heads"],
    num_layers=CONFIG["num_layers"],
    num_classes=CONFIG["num_classes"],
    dropout=CONFIG["dropout"]
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n✅ Model Architecture:")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

# =============================================================================
# CELL 5: Training Setup
# =============================================================================

# Loss function with label smoothing for better generalization
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# AdamW optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
    betas=(0.9, 0.999)
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=CONFIG["learning_rate"],
    epochs=CONFIG["epochs"],
    steps_per_epoch=len(train_loader),
    pct_start=0.1,  # Warmup for 10% of training
    anneal_strategy='cos'
)

# =============================================================================
# CELL 6: Training Loop
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, scheduler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(y_batch).sum().item()
        total += y_batch.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.1f}%'})
    
    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += y_batch.size(0)
    
    return total_loss / len(loader), 100. * correct / total


# Training
print("\n" + "=" * 80)
print("🚀 TRAINING STARTED")
print("=" * 80)

best_val_acc = 0
patience = 15
patience_counter = 0

for epoch in range(CONFIG["epochs"]):
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion)
    
    # Log
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
          f"LR: {current_lr:.2e}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'config': CONFIG,
            'id_to_label': id_to_label,
            'label_to_id': label_to_id,
            'input_dim': CONFIG['input_dim'],
            'num_classes': CONFIG['num_classes'],
        }, "best_model.pth")
        
        print(f"  💾 Saved! (Best Val Acc: {best_val_acc:.2f}%)")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            print(f"   Best validation accuracy: {best_val_acc:.2f}%")
            break

print("\n" + "=" * 80)
print(f"🏆 TRAINING COMPLETE!")
print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
print("=" * 80)

# =============================================================================
# CELL 7: Save Outputs
# =============================================================================

# Save label mapping
with open("label_mapping.json", "w") as f:
    json.dump({
        "id_to_label": {str(k): v for k, v in id_to_label.items()},
        "label_to_id": {v: k for k, v in label_to_id.items()},
        "num_classes": CONFIG["num_classes"],
        "input_dim": CONFIG["input_dim"],
        "best_val_acc": float(best_val_acc),
    }, f, indent=2)

print("\n" + "=" * 80)
print("📦 OUTPUT FILES READY (in 'Output' tab):")
print("=" * 80)
print("  ✅ best_model.pth")
print("  ✅ label_mapping.json")
print("\n🎯 NEXT STEPS:")
print("  1. Download both files from the Output tab")
print("  2. Copy to your project: backend/checkpoints/")
print("  3. Restart your backend server")
print("  4. Test with real sign recognition!")
print("=" * 80)
