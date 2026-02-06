"""
SignSpeak Sign Language Recognition - Training Notebook
========================================================
This notebook trains an LSTM model on collected sign language gesture data.

SETUP INSTRUCTIONS:
1. Upload this file to Kaggle as a new notebook
2. Add your private dataset: "username/signspeak-training-data"
3. Replace DATASET_PATH below with your dataset slug
4. Set Accelerator to GPU T4 x2
5. Run all cells!
"""

import json
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION - REPLACE THIS WITH YOUR DATASET SLUG
# ============================================================================
DATASET_PATH = "/kaggle/input/manual-isl-cache"  

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3
SEQUENCE_LENGTH = 32  # Fixed sequence length

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# DATA LOADING
# ============================================================================

def extract_landmarks_from_sample(sample):
    """Extract and flatten all landmarks from a single sample."""
    features = []
    
    # Pose landmarks (33 points × 4 values = 132 features)
    if 'pose' in sample:
        pose = np.array(sample['pose']).flatten()
        features.append(pose)
    
    # Left hand landmarks (21 points × 3 values = 63 features)
    if 'leftHand' in sample:
        left_hand = np.array(sample['leftHand']).flatten()
        features.append(left_hand)
    
    # Right hand landmarks (21 points × 3 values = 63 features)
    if 'rightHand' in sample:
        right_hand = np.array(sample['rightHand']).flatten()
        features.append(right_hand)
    
    # Face landmarks (468 points × 3 values = 1404 features)
    if 'face' in sample:
        face = np.array(sample['face']).flatten()
        features.append(face)
    
    return np.concatenate(features)

def load_dataset(dataset_path):
    """Load all JSON files and extract features."""
    X = []  # Features
    y = []  # Labels
    
    # Find all JSON cache files
    json_files = glob.glob(os.path.join(dataset_path, "*_cache.json"))
    
    if not json_files:
        raise ValueError(f"No JSON cache files found in {dataset_path}")
    
    print(f"Found {len(json_files)} word files")
    
    for json_file in json_files:
        # Extract word from filename (e.g., "i_cache.json" -> "i")
        filename = os.path.basename(json_file)
        word = filename.replace('_cache.json', '')
        
        print(f"Loading: {word}")
        
        # Load JSON file
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Process each sample
        for sample_frames in data['samples']:
            # Each sample should have 32 frames
            if len(sample_frames) != SEQUENCE_LENGTH:
                print(f"Warning: {word} has {len(sample_frames)} frames, expected {SEQUENCE_LENGTH}")
                continue
            
            # Extract features from each frame
            frame_features = []
            for frame in sample_frames:
                features = extract_landmarks_from_sample(frame)
                frame_features.append(features)
            
            X.append(np.array(frame_features))
            y.append(word)
    
    return np.array(X), np.array(y)

# ============================================================================
# DATASET CLASS
# ============================================================================

class SignLanguageDataset(Dataset):
    def __init__(self, X, y, label_encoder):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(label_encoder.transform(y))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(SignLanguageLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take output from last time step
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """Train the model and track metrics."""
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
            print(f"✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
    
    return train_losses, val_losses, val_accuracies

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("=" * 70)
print("SignSpeak Training Pipeline")
print("=" * 70)

# 1. Load data
print("\n[1/6] Loading dataset...")
X, y = load_dataset(DATASET_PATH)
print(f"Loaded {len(X)} samples")
print(f"Sequence shape: {X[0].shape}")
print(f"Feature dimension: {X[0].shape[1]}")

# 2. Encode labels
print("\n[2/6] Encoding labels...")
label_encoder = LabelEncoder()
label_encoder.fit(y)
vocabulary = label_encoder.classes_
num_classes = len(vocabulary)

print(f"Vocabulary ({num_classes} words): {list(vocabulary)}")

# Save vocabulary
with open('vocabulary.json', 'w') as f:
    json.dump(list(vocabulary), f, indent=2)
print("✓ Vocabulary saved to vocabulary.json")

# 3. Split data
print("\n[3/6] Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} samples | Val: {len(X_val)} samples")

# 4. Create data loaders
print("\n[4/6] Creating data loaders...")
train_dataset = SignLanguageDataset(X_train, y_train, label_encoder)
val_dataset = SignLanguageDataset(X_val, y_val, label_encoder)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 5. Initialize model
print("\n[5/6] Initializing model...")
input_size = X.shape[2]  # Feature dimension
model = SignLanguageLSTM(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=num_classes,
    dropout=DROPOUT
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 6. Train model
print("\n[6/6] Training model...")
print("=" * 70)
train_losses, val_losses, val_accuracies = train_model(
    model, train_loader, val_loader, criterion, optimizer, EPOCHS
)

# ============================================================================
# RESULTS & VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("Training Complete!")
print("=" * 70)
print(f"Best Validation Accuracy: {max(val_accuracies):.2f}%")

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
ax1.plot(train_losses, label='Train Loss', linewidth=2)
ax1.plot(val_losses, label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy curve
ax2.plot(val_accuracies, label='Val Accuracy', color='green', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Validation Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
print("\n✓ Training curves saved to training_curves.png")

# Save final model package
print("\n" + "=" * 70)
print("Saving Model Package...")
print("=" * 70)

# Load best model
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Save complete model package
torch.save({
    'model_state_dict': model.state_dict(),
    'vocabulary': list(vocabulary),
    'input_size': input_size,
    'hidden_size': HIDDEN_SIZE,
    'num_layers': NUM_LAYERS,
    'num_classes': num_classes,
    'best_val_acc': checkpoint['val_acc'],
    'label_encoder_classes': label_encoder.classes_.tolist()
}, 'sign_model.pth')

print("✓ Model saved to: sign_model.pth")
print("✓ Vocabulary saved to: vocabulary.json")

print("\n📥 Download these files:")
print("   1. sign_model.pth (for PyTorch inference)")
print("   2. vocabulary.json (list of words)")
print("   3. training_curves.png (performance visualization)")

print("\n" + "=" * 70)
print("📤 Upload to Kaggle Models")
print("=" * 70)
print("To update your model:")
print("1. Download sign_model.pth and vocabulary.json from output above")
print("2. Go to: kaggle.com/models/kathiravankoffl/signspeak-model")
print("3. Click 'New Version' and upload both files")
print(f"4. Version notes: Trained on {num_classes} words: {', '.join(vocabulary)}. Val Acc: {max(val_accuracies):.2f}%")

print("\n" + "=" * 70)
print("🎉 Training Pipeline Complete!")
print("=" * 70)


