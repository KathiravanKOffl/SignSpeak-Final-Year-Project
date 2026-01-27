# SignSpeak - WLASL Alphabet Training (FIXED FOR OVERFITTING)
# Lightweight model for 25-class alphabet recognition

"""
PROBLEM: Previous model had 19M params, causing severe overfitting
- Train: 94% | Val: 18%  ← BAD

SOLUTION: Simpler model + heavy regularization
- Target: Train ~70% | Val ~60%+ ← Much better generalization
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
print("🚀 SignSpeak - WLASL FIXED (Anti-Overfitting)")
print("=" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✅ Device: {device}")

# =============================================================================
# Configuration - FIXED FOR SMALL DATASET
# =============================================================================

CONFIG = {
    # Model - MUCH SMALLER
    "hidden_dim": 128,      # Was 512 → Now 128
    "num_layers": 2,        # Was 6 → Now 2
    "dropout": 0.5,         # Was 0.2 → Now 0.5 (heavy dropout)
    
    # Training
    "batch_size": 64,       # Smaller batches = more regularization
    "epochs": 100,
    "learning_rate": 1e-3,  # Higher LR for simpler model
    "weight_decay": 0.05,   # Was 0.001 → Now 0.05 (strong L2)
    "label_smoothing": 0.2, # Was 0.1 → Now 0.2
    
    # Early stopping on VAL LOSS (not accuracy)
    "patience": 15,
    
    # Augmentation - MORE AGGRESSIVE
    "aug_rotation": 15,
    "aug_scale": 0.15,
    "aug_noise": 0.02,
}

# =============================================================================
# Load Data
# =============================================================================

print("\n" + "=" * 80)
print("📥 LOADING DATASET")
print("=" * 80)

data_path = Path("/kaggle/input/land-mark-holistic-featuresfor-wlasl")
if not data_path.exists():
    for p in Path("/kaggle/input").rglob("x_output*.npy"):
        data_path = p.parent
        break

x_file = list(data_path.glob("x_output*.npy"))[0]
y_file = list(data_path.glob("y_output*.npy"))[0]

X_raw = np.load(x_file).astype(np.float32)
y_raw = np.load(y_file)

print(f"Shape: {X_raw.shape}")

# Decode labels
if y_raw.dtype.kind == 'S':
    y_labels = np.array([l.decode('utf-8') if isinstance(l, bytes) else str(l) for l in y_raw])
else:
    y_labels = y_raw.astype(str)

unique_labels = sorted(set(y_labels))
label_to_id = {l: i for i, l in enumerate(unique_labels)}
id_to_label = {i: l for i, l in enumerate(unique_labels)}
y = np.array([label_to_id[l] for l in y_labels])

num_classes = len(unique_labels)
print(f"Classes: {num_classes}")
print(f"Samples per class: ~{len(y)//num_classes}")

# Clean
X_raw = np.nan_to_num(X_raw, 0)

# =============================================================================
# Augmentation - STRONGER
# =============================================================================

def augment(X):
    """Aggressive augmentation to prevent overfitting"""
    # Random rotation
    if np.random.random() < 0.5:
        angle = np.random.uniform(-15, 15) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        X = X.copy()
        for t in range(X.shape[0]):
            X[t] = X[t] @ rot.T
    
    # Random scale
    if np.random.random() < 0.5:
        scale = np.random.uniform(0.85, 1.15)
        X = X * scale
    
    # Random noise
    if np.random.random() < 0.5:
        noise = np.random.normal(0, 0.02, X.shape)
        X = X + noise
    
    # Random temporal shift (shift frames)
    if np.random.random() < 0.3:
        shift = np.random.randint(-3, 4)
        X = np.roll(X, shift, axis=0)
    
    return X.astype(np.float32)

# =============================================================================
# Dataset
# =============================================================================

class ASLDataset(Dataset):
    def __init__(self, X, y, augment_fn=None):
        self.X = X
        self.y = y
        self.augment_fn = augment_fn
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment_fn:
            x = self.augment_fn(x)
        # Average over time: (30, 75, 2) → (150,)
        x = x.mean(axis=0).flatten()
        return torch.FloatTensor(x), torch.LongTensor([self.y[idx]])[0]

# Split with more validation data
X_train, X_val, y_train, y_val = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)

train_loader = DataLoader(
    ASLDataset(X_train, y_train, augment), 
    batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0
)
val_loader = DataLoader(
    ASLDataset(X_val, y_val, None), 
    batch_size=CONFIG["batch_size"], num_workers=0
)

print(f"Train: {len(X_train)} | Val: {len(X_val)}")

# =============================================================================
# Model - SIMPLE MLP (Much smaller)
# =============================================================================

class SimpleClassifier(nn.Module):
    """
    Lightweight MLP for alphabet recognition
    ~100K params instead of 19M
    """
    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        super().__init__()
        
        self.net = nn.Sequential(
            # Input layer
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Hidden layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

input_dim = 75 * 2  # 150 features (averaged over time)
model = SimpleClassifier(
    input_dim=input_dim,
    hidden_dim=CONFIG["hidden_dim"],
    num_classes=num_classes,
    dropout=CONFIG["dropout"]
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"\n✅ Model: {num_params:,} parameters (was 19M, now ~50K)")

# =============================================================================
# Training
# =============================================================================

criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

print("\n" + "=" * 80)
print("🚀 TRAINING (Anti-Overfitting Mode)")
print("=" * 80)

best_val_loss = float('inf')
best_val_acc = 0
patience_counter = 0
start_time = time.time()

for epoch in range(CONFIG["epochs"]):
    # Train
    model.train()
    train_loss, correct, total = 0, 0, 0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        out = model(X_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        correct += (out.argmax(1) == y_b).sum().item()
        total += y_b.size(0)
    
    train_acc = 100. * correct / total
    avg_train_loss = train_loss / len(train_loader)
    
    # Validate
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
    
    scheduler.step(avg_val_loss)
    elapsed = (time.time() - start_time) / 60
    
    # Check overfitting
    gap = train_acc - val_acc
    overfit_warning = "⚠️ OVERFIT" if gap > 20 else ""
    
    print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
          f"Train: {avg_train_loss:.3f} ({train_acc:.1f}%) | "
          f"Val: {avg_val_loss:.3f} ({val_acc:.1f}%) | "
          f"Gap: {gap:.1f}% {overfit_warning}")
    
    # Early stopping on VAL LOSS (not accuracy!)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_val_acc = val_acc
        patience_counter = 0
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': CONFIG,
            'id_to_label': id_to_label,
            'input_dim': input_dim,
            'num_classes': num_classes,
            'val_acc': val_acc,
            'val_loss': avg_val_loss
        }, "best_model.pth")
        print(f"  💾 Saved! Val Loss: {best_val_loss:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= CONFIG["patience"]:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break

print(f"\n🏆 FINAL: Best Val Acc = {best_val_acc:.2f}%, Val Loss = {best_val_loss:.4f}")

# Save mapping
with open("label_mapping.json", "w") as f:
    json.dump({
        "id_to_label": {str(k): v for k, v in id_to_label.items()},
        "label_to_id": label_to_id,
        "num_classes": num_classes,
        "input_dim": input_dim,
        "best_val_acc": float(best_val_acc),
        "dataset": "WLASL-Alphabet-25",
        "model_type": "SimpleMLP"
    }, f, indent=2)

print("\n📦 Done! Download: best_model.pth + label_mapping.json")
