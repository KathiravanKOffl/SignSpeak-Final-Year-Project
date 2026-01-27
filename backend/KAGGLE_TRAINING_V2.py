# SignSpeak Kaggle Training V2 - Fixed for WLASL Holistic Features
# This version handles MULTIPLE NPY FILE FORMATS properly

"""
## SETUP INSTRUCTIONS:

1. Create New Kaggle Notebook
2. Click "Add data" → Search "land-mark-holistic-features-WLASL" → Add
   (Alternative: "WLASL holistic" or "MediaPipe WLASL")
3. Settings → Accelerator → GPU P100 or T4
4. Copy ALL this code → Paste → Run All
5. Training runs overnight (4-8 hours)
6. Download best_model.pth + label_mapping.json from Output tab
"""

# =============================================================================
# CELL 1: Setup
# =============================================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")

# Config
CONFIG = {
    "input_dim": 225,  # Will be adjusted based on dataset
    "hidden_dim": 256,
    "num_heads": 8,
    "num_layers": 4,
    "num_classes": 100,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-4,
    "dropout": 0.3,
    "max_frames": 64,  # Fixed frame length for padding/truncating
}

# =============================================================================
# CELL 2: Enhanced Data Loader
# =============================================================================

def load_npy_with_multiple_formats(npy_file):
    """
    Handle different NPY file formats:
    - (frames, landmarks, 3) → flatten to (frames * landmarks * 3)
    - (landmarks, 3) → flatten to (landmarks * 3)
    - (features,) → use as is
    - (frames, features) → average over frames
    """
    try:
        data = np.load(npy_file, allow_pickle=True)
        
        # Handle object arrays (sometimes NPY files are saved with allow_pickle)
        if data.dtype == np.object_:
            # Try to extract the actual array
            if len(data.shape) == 0:  # Scalar object
                data = data.item()
                if isinstance(data, np.ndarray):
                    data = data.astype(np.float32)
            elif len(data) > 0:
                # Array of objects - try first element
                first = data.flat[0]
                if isinstance(first, np.ndarray):
                    return first.astype(np.float32).flatten()
        
        # Now handle regular numeric arrays
        data = data.astype(np.float32)
        
        if len(data.shape) == 1:
            # Already flat
            return data
        
        elif len(data.shape) == 2:
            # Could be (frames, features) or (landmarks, 3)
            if data.shape[1] == 3:
                # Probably (landmarks, 3) - flatten
                return data.flatten()
            else:
                # Probably (frames, features) - average over frames
                return data.mean(axis=0)
        
        elif len(data.shape) == 3:
            # (frames, landmarks, 3) - flatten to (frames, landmarks*3) then average
            frames, landmarks, coords = data.shape
            # Flatten last two dimensions
            reshaped = data.reshape(frames, landmarks * coords)
            # Average over frames
            return reshaped.mean(axis=0)
        
        else:
            # Unknown shape - just flatten everything
            return data.flatten()
    
    except Exception as e:
        return None


def smart_load_wlasl_holistic():
    """Load WLASL holistic features with intelligent format detection"""
    base_path = Path("/kaggle/input")
    
    # Find the dataset folder
    possible_patterns = ['holistic', 'landmark', 'wlasl', 'mediapipe']
    
    data_path = None
    for folder in base_path.iterdir():
        if folder.is_dir():
            folder_lower = folder.name.lower()
            if any(pattern in folder_lower for pattern in possible_patterns):
                # Prefer folders with "holistic" or "landmark"
                if 'holistic' in folder_lower or 'landmark' in folder_lower:
                    data_path = folder
                    break
                elif data_path is None:
                    data_path = folder
    
    if not data_path:
        print("⚠️ No WLASL/holistic dataset found!")
        return None, None
    
    print(f"📂 Found: {data_path}")
    print(f"   Scanning files...")
    
    X_list = []
    y_list = []
    
    # Try MULTIPLE loading strategies
    
    # Strategy 1: Look for NPY files organized by class folders
    npy_files = list(data_path.rglob("*.npy"))
    print(f"   Found {len(npy_files)} NPY files")
    
    if npy_files:
        print(f"   Loading from NPY files...")
        
        # Check if organized by folders
        labels_from_folders = {}
        for npy in npy_files:
            # Try to get label from parent folder
            parts = npy.relative_to(data_path).parts
            if len(parts) >= 2:
                potential_label = parts[-2]  # Parent folder
            else:
                potential_label = npy.stem  # Filename without extension
            
            labels_from_folders.setdefault(potential_label, []).append(npy)
        
        # If we have multiple files per label, use folder structure
        if len(labels_from_folders) > 1 and max(len(v) for v in labels_from_folders.values()) > 1:
            print(f"   Detected folder structure: {len(labels_from_folders)} classes")
            
            # Load samples from each class
            for label, files in tqdm(labels_from_folders.items(), desc="Loading classes"):
                # Limit samples per class to balance
                for npy_file in files[:200]:  # Max 200 per class
                    features = load_npy_with_multiple_formats(npy_file)
                    if features is not None and len(features) >= 50:
                        X_list.append(features)
                        y_list.append(str(label))
        
        # Otherwise, just load all NPY files
        else:
            print(f"   Loading individual NPY files...")
            sample_shapes = set()
            
            for npy_file in tqdm(npy_files[:5000], desc="Loading NPY"):
                features = load_npy_with_multiple_formats(npy_file)
                
                if features is not None and len(features) >= 50:
                    X_list.append(features)
                    
                    # Try to get label from filename or folder
                    parts = npy_file.relative_to(data_path).parts
                    if len(parts) >= 2:
                        label = parts[-2]
                    else:
                        # Use stem or create synthetic label
                        label = npy_file.stem.split('_')[0] if '_' in npy_file.stem else npy_file.stem
                    
                    y_list.append(str(label))
                    
                    # Track shapes for debugging
                    sample_shapes.add(len(features))
            
            print(f"   Loaded {len(X_list)} samples")
            print(f"   Feature dimensions found: {sorted(sample_shapes)[:5]}")
    
    # Strategy 2: Check for CSV/Parquet files
    if len(X_list) == 0:
        csv_files = list(data_path.rglob("*.csv"))
        parquet_files = list(data_path.rglob("*.parquet"))
        
        print(f"   Trying CSV ({len(csv_files)}) and Parquet ({len(parquet_files)}) files...")
        
        for pf in parquet_files[:50]:
            try:
                df = pd.read_parquet(pf)
                # Look for label column
                label_cols = [c for c in df.columns if c.lower() in ['label', 'sign', 'gloss', 'word', 'class', 'target']]
                
                if label_cols:
                    label_col = label_cols[0]
                    feature_cols = [c for c in df.columns if c != label_col]
                    
                    for _, row in df.iterrows():
                        features = row[feature_cols].values.astype(np.float32)
                        if len(features) >= 50:
                            X_list.append(features)
                            y_list.append(str(row[label_col]))
                else:
                    # No label column - use filename
                    feature_cols = [c for c in df.columns if df[c].dtype in ['float64', 'float32', 'int64']]
                    label = pf.stem
                    
                    for _, row in df.iterrows():
                        features = row[feature_cols].values.astype(np.float32)
                        if len(features) >= 50:
                            X_list.append(features)
                            y_list.append(label)
            except Exception as e:
                pass
    
    if len(X_list) == 0:
        return None, None
    
    print(f"✅ Loaded {len(X_list)} samples from {len(set(y_list))} classes")
    
    # Make all features same length
    if X_list:
        min_len = min(len(x) for x in X_list)
        max_len = max(len(x) for x in X_list)
        
        # Use median length or truncate to reasonable size
        target_len = min(300, int(np.median([len(x) for x in X_list])))
        
        print(f"   Feature length range: {min_len} - {max_len}")
        print(f"   Standardizing to: {target_len}")
        
        # Pad or truncate
        X_standardized = []
        for x in X_list:
            if len(x) >= target_len:
                X_standardized.append(x[:target_len])
            else:
                # Pad with zeros
                padded = np.zeros(target_len, dtype=np.float32)
                padded[:len(x)] = x
                X_standardized.append(padded)
        
        return np.array(X_standardized), np.array(y_list)
    
    return None, None


# =============================================================================
# CELL 3: Load Data
# =============================================================================

print("\n" + "="*60)
print("📥 LOADING DATASETS")
print("="*60 + "\n")

X, y_labels = smart_load_wlasl_holistic()

if X is None:
    print("\n⚠️ FAILED TO LOAD DATA!")
    print("Running diagnostic to help you...")
    print("\nPlease:")
    print("1. Check that you added the correct dataset")
    print("2. Verify dataset name contains 'holistic' or 'wlasl'")
    print("3. Look at the 'Data' tab on the right to see folder structure")
    raise RuntimeError("No data loaded - check dataset is attached!")

# Encode labels
unique_labels = sorted(set(y_labels))
label_to_id = {l: i for i, l in enumerate(unique_labels)}
id_to_label = {i: l for i, l in enumerate(unique_labels)}
y = np.array([label_to_id[l] for l in y_labels])

CONFIG["num_classes"] = len(unique_labels)
CONFIG["input_dim"] = X.shape[1]

print(f"\n✅ Dataset Ready:")
print(f"   Samples: {len(X)}")
print(f"   Classes: {CONFIG['num_classes']}")
print(f"   Features: {CONFIG['input_dim']}")
print(f"   Sample labels: {list(unique_labels)[:10]}")

# =============================================================================
# CELL 4: Dataset & Model
# =============================================================================

class SignDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_loader = DataLoader(SignDataset(X_train, y_train), batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
val_loader = DataLoader(SignDataset(X_val, y_val), batch_size=CONFIG["batch_size"], num_workers=2)

print(f"✅ Train: {len(X_train)}, Val: {len(X_val)}")

# Hybrid CNN-Transformer Model
class HybridSignTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, num_classes):
        super().__init__()
        
        # Spatial Encoder
        self.spatial = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim*4, dropout=CONFIG["dropout"],
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        x = self.spatial(x)
        x = x.unsqueeze(1)  # Add sequence dim
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.classifier(x)

model = HybridSignTransformer(
    CONFIG["input_dim"], CONFIG["hidden_dim"],
    CONFIG["num_heads"], CONFIG["num_layers"], CONFIG["num_classes"]
).to(device)

print(f"✅ Model: {sum(p.numel() for p in model.parameters()):,} parameters")

# =============================================================================
# CELL 5: Training
# =============================================================================

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

def train_epoch(model, loader):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        out = model(X_b)
        loss = criterion(out, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item()
        correct += (out.argmax(1) == y_b).sum().item()
        total += y_b.size(0)
    return loss_sum/len(loader), 100*correct/total

def validate(model, loader):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            out = model(X_b)
            loss_sum += criterion(out, y_b).item()
            correct += (out.argmax(1) == y_b).sum().item()
            total += y_b.size(0)
    return loss_sum/len(loader), 100*correct/total

# Train
print("\n" + "="*60)
print("🚀 TRAINING")
print("="*60)

best_acc = 0
patience = 10
patience_counter = 0

for epoch in range(CONFIG["epochs"]):
    train_loss, train_acc = train_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)
    scheduler.step()
    
    print(f"Epoch {epoch+1:2d}/{CONFIG['epochs']} | "
          f"Train: {train_loss:.4f} ({train_acc:.1f}%) | "
          f"Val: {val_loss:.4f} ({val_acc:.1f}%)")
    
    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": CONFIG,
            "id_to_label": id_to_label,
            "input_dim": CONFIG["input_dim"],
            "num_classes": CONFIG["num_classes"],
        }, "best_model.pth")
        print(f"  💾 Saved! (Best: {best_acc:.1f}%)")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break

print(f"\n🏆 Training Complete! Best Validation Accuracy: {best_acc:.1f}%")

# =============================================================================
# CELL 6: Save Outputs
# =============================================================================

# Save label mapping
with open("label_mapping.json", "w") as f:
    json.dump({
        "id_to_label": {str(k): v for k, v in id_to_label.items()},
        "label_to_id": {v: k for k, v in id_to_label.items()},
        "num_classes": CONFIG["num_classes"],
        "input_dim": CONFIG["input_dim"]
    }, f, indent=2)

print("\n" + "="*60)
print("📦 OUTPUT FILES (Download from 'Output' tab):")
print("="*60)
print("  ✅ best_model.pth")
print("  ✅ label_mapping.json")
print("\n🚀 NEXT STEPS:")
print("  1. Download both files")
print(f"  2. Copy to: backend/checkpoints/")
print("  3. Restart your backend server")
print("="*60)
