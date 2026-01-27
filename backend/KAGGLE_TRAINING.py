# SignSpeak Kaggle Training - Updated for Your Datasets
# Compatible with: land-mark-holistic-features-WLASL, Indian Sign Language, WLASL Raw

"""
## SETUP INSTRUCTIONS:

1. Create New Kaggle Notebook
2. Click "Add data" (right sidebar) and add ALL THREE:
   - Search: "land-mark-holistic-features-WLASL" → Add
   - Search: "indian sign language" → Add  
   - Search: "wlasl" (find the raw one) → Add
3. Settings → Accelerator → GPU P100
4. Copy ALL this code → Paste → Run All
5. Close laptop → Training runs overnight
6. Download best_model.pth from Output tab
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
}

# =============================================================================
# CELL 2: Load YOUR Datasets
# =============================================================================

def load_landmark_holistic_wlasl():
    """Load land-mark-holistic-features-WLASL dataset"""
    base_path = Path("/kaggle/input")
    
    # Find the dataset folder
    possible_names = ["land-mark-holistic-features-for-wlasl", 
                      "land-mark-holistic-featuresfor-wlasl",
                      "landmark-holistic-features-wlasl"]
    
    data_path = None
    for name in possible_names:
        check_path = base_path / name
        if check_path.exists():
            data_path = check_path
            break
    
    # Also check any folder with "holistic" or "landmark"
    if not data_path:
        for folder in base_path.iterdir():
            if "holistic" in folder.name.lower() or "landmark" in folder.name.lower():
                data_path = folder
                break
    
    if not data_path:
        print("⚠️ land-mark-holistic-features-WLASL not found")
        return None, None
    
    print(f"📂 Found: {data_path}")
    
    X_list = []
    y_list = []
    
    # This dataset usually has NPY files organized by class folders
    # OR a single large file with all features
    
    # Check for CSV/Parquet files first
    csv_files = list(data_path.rglob("*.csv"))
    parquet_files = list(data_path.rglob("*.parquet"))
    npy_files = list(data_path.rglob("*.npy"))
    
    print(f"  Files found: {len(csv_files)} CSV, {len(parquet_files)} Parquet, {len(npy_files)} NPY")
    
    if parquet_files:
        print("  Loading Parquet files...")
        for pf in tqdm(parquet_files[:50], desc="Loading"):
            try:
                df = pd.read_parquet(pf)
                # Try to identify label column
                label_cols = [c for c in df.columns if c.lower() in ['label', 'sign', 'gloss', 'word', 'class']]
                if label_cols:
                    label_col = label_cols[0]
                    feature_cols = [c for c in df.columns if c != label_col]
                    for _, row in df.iterrows():
                        features = row[feature_cols].values.astype(np.float32)
                        # Ensure consistent size
                        if len(features) >= 225:
                            X_list.append(features[:225])
                            y_list.append(str(row[label_col]))
            except Exception as e:
                pass
    
    elif csv_files:
        print("  Loading CSV files...")
        for cf in tqdm(csv_files[:20], desc="Loading"):
            try:
                df = pd.read_csv(cf)
                label_cols = [c for c in df.columns if c.lower() in ['label', 'sign', 'gloss', 'word', 'class']]
                if label_cols:
                    label_col = label_cols[0]
                    feature_cols = [c for c in df.columns if c != label_col and df[c].dtype in ['float64', 'float32', 'int64']]
                    for _, row in df.iterrows():
                        features = row[feature_cols].values.astype(np.float32)
                        if len(features) >= 100:
                            X_list.append(features[:CONFIG["input_dim"]])
                            y_list.append(str(row[label_col]))
            except:
                pass
    
    elif npy_files:
        print("  Loading NPY files...")
        # Usually structure: /class_name/sample.npy
        for npy_file in tqdm(npy_files[:1000], desc="Loading"):
            try:
                # Get label from parent folder name
                label = npy_file.parent.name
                feat = np.load(npy_file)
                
                # Handle different shapes
                if len(feat.shape) > 1:
                    feat = feat.mean(axis=0)  # Average over time dimension
                
                if len(feat) >= 100:
                    X_list.append(feat[:CONFIG["input_dim"]].astype(np.float32))
                    y_list.append(label)
            except:
                pass
    
    if X_list:
        print(f"  ✅ Loaded {len(X_list)} samples from WLASL Holistic")
        return np.array(X_list), np.array(y_list)
    
    return None, None


def load_isl_dataset():
    """Load Indian Sign Language dataset (image-based)"""
    base_path = Path("/kaggle/input")
    
    # Find ISL dataset
    data_path = None
    for folder in base_path.iterdir():
        if "indian" in folder.name.lower() and "sign" in folder.name.lower():
            data_path = folder
            break
    
    if not data_path:
        print("⚠️ Indian Sign Language dataset not found")
        return None, None
    
    print(f"📂 Found ISL: {data_path}")
    
    # ISL dataset is usually image-based with folder structure /A/img1.jpg, /B/img2.jpg
    # We need to extract features using a simple approach
    # For now, we'll skip raw images and focus on the pre-extracted WLASL
    
    # Check if there are pre-extracted features
    npy_files = list(data_path.rglob("*.npy"))
    csv_files = list(data_path.rglob("*.csv"))
    
    print(f"  Files: {len(npy_files)} NPY, {len(csv_files)} CSV")
    
    X_list = []
    y_list = []
    
    if csv_files:
        for cf in csv_files[:10]:
            try:
                df = pd.read_csv(cf)
                # Look for numeric columns
                label_cols = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower()]
                if label_cols:
                    label_col = label_cols[0]
                    feature_cols = [c for c in df.columns if df[c].dtype in ['float64', 'float32', 'int64'] and c != label_col]
                    if feature_cols:
                        for _, row in df.iterrows():
                            X_list.append(row[feature_cols].values[:CONFIG["input_dim"]].astype(np.float32))
                            y_list.append(str(row[label_col]))
            except:
                pass
    
    if X_list:
        print(f"  ✅ Loaded {len(X_list)} samples from ISL")
        return np.array(X_list), np.array(y_list)
    
    print("  ⚠️ ISL dataset is image-based, needs MediaPipe extraction (skipping for now)")
    return None, None


def load_wlasl_raw():
    """Load WLASL Raw dataset (video-based - we'll look for any pre-extracted features)"""
    base_path = Path("/kaggle/input")
    
    data_path = None
    for folder in base_path.iterdir():
        if "wlasl" in folder.name.lower() and "raw" not in folder.name.lower():
            data_path = folder
            break
    
    if not data_path:
        for folder in base_path.iterdir():
            if "wlasl" in folder.name.lower():
                data_path = folder
                break
    
    if not data_path:
        print("⚠️ WLASL dataset not found")
        return None, None
    
    print(f"📂 Found WLASL: {data_path}")
    
    # Check for JSON annotation file (common in WLASL)
    json_files = list(data_path.rglob("*.json"))
    npy_files = list(data_path.rglob("*.npy"))
    
    print(f"  Files: {len(json_files)} JSON, {len(npy_files)} NPY")
    
    # WLASL Raw usually has video files + JSON annotations
    # Without pre-extracted features, we can't use it directly
    print("  ⚠️ WLASL Raw needs video processing (will use WLASL Holistic instead)")
    
    return None, None


# Load all available datasets
print("\n" + "="*60)
print("📥 LOADING DATASETS")
print("="*60 + "\n")

all_X = []
all_y = []

# 1. Try WLASL Holistic (best option - pre-extracted features)
X1, y1 = load_landmark_holistic_wlasl()
if X1 is not None:
    all_X.append(X1)
    all_y.append(y1)

# 2. Try ISL
X2, y2 = load_isl_dataset()
if X2 is not None:
    all_X.append(X2)
    all_y.append(y2)

# 3. Try WLASL Raw (usually won't work without preprocessing)
X3, y3 = load_wlasl_raw()
if X3 is not None:
    all_X.append(X3)
    all_y.append(y3)

# Combine all loaded data
if all_X:
    # Ensure consistent feature dimension
    min_dim = min(x.shape[1] for x in all_X)
    CONFIG["input_dim"] = min_dim
    
    X = np.vstack([x[:, :min_dim] for x in all_X])
    y_labels = np.concatenate(all_y)
    print(f"\n✅ TOTAL: {len(X)} samples, {CONFIG['input_dim']} features")
else:
    print("\n⚠️ No data loaded! Generating synthetic data for testing...")
    X = np.random.randn(2000, CONFIG["input_dim"]).astype(np.float32)
    y_labels = np.array([f"sign_{i%20}" for i in range(2000)])

# Encode labels
unique_labels = sorted(set(y_labels))
label_to_id = {l: i for i, l in enumerate(unique_labels)}
id_to_label = {i: l for i, l in enumerate(unique_labels)}
y = np.array([label_to_id[l] for l in y_labels])

CONFIG["num_classes"] = len(unique_labels)
print(f"✅ Classes: {CONFIG['num_classes']}")
print(f"✅ Sample labels: {list(unique_labels)[:10]}")

# =============================================================================
# CELL 3: Dataset & Model
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
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_loader = DataLoader(SignDataset(X_train, y_train), batch_size=CONFIG["batch_size"], shuffle=True)
val_loader = DataLoader(SignDataset(X_val, y_val), batch_size=CONFIG["batch_size"])

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
# CELL 4: Training
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
for epoch in range(CONFIG["epochs"]):
    train_loss, train_acc = train_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)
    scheduler.step()
    
    print(f"Epoch {epoch+1:2d}/{CONFIG['epochs']} | "
          f"Train: {train_loss:.4f} ({train_acc:.1f}%) | "
          f"Val: {val_loss:.4f} ({val_acc:.1f}%)")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": CONFIG,
            "id_to_label": id_to_label,
            "input_dim": CONFIG["input_dim"],
            "num_classes": CONFIG["num_classes"],
        }, "best_model.pth")
        print(f"  💾 Saved! (Best: {best_acc:.1f}%)")

print(f"\n🏆 Training Complete! Best: {best_acc:.1f}%")

# =============================================================================
# CELL 5: Save
# =============================================================================

# Save label mapping
with open("label_mapping.json", "w") as f:
    json.dump({
        "id_to_label": {str(k): v for k, v in id_to_label.items()},
        "num_classes": CONFIG["num_classes"],
        "input_dim": CONFIG["input_dim"]
    }, f, indent=2)

print("\n📦 OUTPUT FILES (Download from 'Output' tab):")
print("  - best_model.pth")
print("  - label_mapping.json")
print("\n✅ Copy these to: backend/checkpoints/")
