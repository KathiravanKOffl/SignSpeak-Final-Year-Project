# SignSpeak - WLASL Training (BALANCED - Temporal LSTM)
# Properly sized model with temporal processing

"""
Previous attempts:
- 19M params → 94% train, 18% val (OVERFITTING)
- 40K params → 15% train, 16% val (UNDERFITTING)

This version:
- ~500K params with LSTM for temporal processing
- Target: 50-60% train, 40-50% val (good generalization)
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
print("🚀 SignSpeak - TEMPORAL LSTM MODEL")
print("=" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✅ Device: {device}")

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # Model - BALANCED
    "hidden_dim": 256,
    "lstm_layers": 2,
    "dropout": 0.4,
    
    # Training
    "batch_size": 64,
    "epochs": 80,
    "learning_rate": 5e-4,
    "weight_decay": 0.01,
    "label_smoothing": 0.15,
    
    # Early stopping
    "patience": 20,
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

print(f"Shape: {X_raw.shape}")  # (11980, 30, 75, 2)

# Reshape: (N, 30, 75, 2) → (N, 30, 150)
N, T, J, C = X_raw.shape
X_raw = X_raw.reshape(N, T, J * C)
print(f"Reshaped: {X_raw.shape}")  # (11980, 30, 150)

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

# Clean NaNs
X_raw = np.nan_to_num(X_raw, 0)

# Normalize per sample
for i in range(len(X_raw)):
    sample = X_raw[i]
    mean = sample.mean()
    std = sample.std() + 1e-8
    X_raw[i] = (sample - mean) / std

# =============================================================================
# Augmentation
# =============================================================================

def augment(X):
    """Moderate augmentation"""
    X = X.copy()
    
    # Time warping (stretch/compress)
    if np.random.random() < 0.3:
        factor = np.random.uniform(0.9, 1.1)
        indices = np.clip(np.arange(30) * factor, 0, 29).astype(int)
        X = X[indices]
    
    # Add noise
    if np.random.random() < 0.4:
        noise = np.random.normal(0, 0.05, X.shape)
        X = X + noise
    
    # Random scale
    if np.random.random() < 0.3:
        scale = np.random.uniform(0.9, 1.1)
        X = X * scale
    
    # Temporal shift
    if np.random.random() < 0.3:
        shift = np.random.randint(-2, 3)
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
        return torch.FloatTensor(x), torch.LongTensor([self.y[idx]])[0]

# Split
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
# Model - Bidirectional LSTM
# =============================================================================

class TemporalLSTM(nn.Module):
    """
    LSTM that processes temporal sequences of landmarks
    Input: (batch, 30, 150) - 30 frames, 150 features each
    """
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, dropout):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, 30, 150)
        x = self.input_proj(x)  # (batch, 30, hidden)
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, 30, hidden*2)
        
        # Use last hidden states from both directions
        # h_n: (num_layers*2, batch, hidden)
        forward_h = h_n[-2]  # Last layer, forward
        backward_h = h_n[-1]  # Last layer, backward
        combined = torch.cat([forward_h, backward_h], dim=1)  # (batch, hidden*2)
        
        return self.classifier(combined)

input_dim = 150  # 75 joints * 2 coordinates
model = TemporalLSTM(
    input_dim=input_dim,
    hidden_dim=CONFIG["hidden_dim"],
    num_classes=num_classes,
    num_layers=CONFIG["lstm_layers"],
    dropout=CONFIG["dropout"]
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"\n✅ Model: {num_params:,} parameters (balanced for this dataset)")

# =============================================================================
# Training
# =============================================================================

criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

print("\n" + "=" * 80)
print("🚀 TRAINING WITH TEMPORAL LSTM")
print("=" * 80)

best_val_acc = 0
best_val_loss = float('inf')
patience_counter = 0
history = []
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    
    scheduler.step()
    elapsed = (time.time() - start_time) / 60
    
    gap = train_acc - val_acc
    status = ""
    if gap > 15:
        status = "⚠️"
    elif val_acc > best_val_acc:
        status = "🎯"
    
    print(f"Epoch {epoch+1:3d}/{CONFIG['epochs']} | "
          f"Train: {avg_train_loss:.3f} ({train_acc:.1f}%) | "
          f"Val: {avg_val_loss:.3f} ({val_acc:.1f}%) | "
          f"Gap: {gap:.1f}% {status}")
    
    history.append({
        'epoch': epoch + 1,
        'train_loss': avg_train_loss,
        'train_acc': train_acc,
        'val_loss': avg_val_loss,
        'val_acc': val_acc
    })
    
    # Save best model (by validation accuracy)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_loss = avg_val_loss
        patience_counter = 0
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': CONFIG,
            'id_to_label': id_to_label,
            'label_to_id': label_to_id,
            'input_dim': input_dim,
            'num_classes': num_classes,
            'val_acc': val_acc,
            'val_loss': avg_val_loss,
            'model_type': 'TemporalLSTM'
        }, "/kaggle/working/best_model.pth")
        print(f"  💾 Saved Best! Val Acc: {best_val_acc:.2f}%")
    else:
        patience_counter += 1
        if patience_counter >= CONFIG["patience"]:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break

print("\n" + "=" * 80)
print(f"🏆 FINAL RESULTS")
print("=" * 80)
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Random Guess Baseline: {100/num_classes:.1f}%")
print(f"Improvement over random: {best_val_acc / (100/num_classes):.1f}x")

# Save mapping
print("\n📝 Saving label_mapping.json...")
with open("/kaggle/working/label_mapping.json", "w") as f:
    json.dump({
        "id_to_label": {str(k): v for k, v in id_to_label.items()},
        "label_to_id": label_to_id,
        "num_classes": num_classes,
        "input_dim": input_dim,
        "seq_len": 30,
        "best_val_acc": float(best_val_acc),
        "dataset": "WLASL-Alphabet-25",
        "model_type": "TemporalLSTM"
    }, f, indent=2)
print("✅ Saved: /kaggle/working/label_mapping.json")

# Save history
print("📝 Saving training_history.json...")
with open("/kaggle/working/training_history.json", "w") as f:
    json.dump(history, f, indent=2)
print("✅ Saved: /kaggle/working/training_history.json")

# List all files in output directory
print("\n📂 Files in /kaggle/working/:")
import os
for f in os.listdir("/kaggle/working/"):
    size = os.path.getsize(f"/kaggle/working/{f}") / (1024*1024)  # MB
    print(f"   - {f} ({size:.2f} MB)")

print("\n📦 Done! Download: best_model.pth + label_mapping.json")
print("=" * 80)
