"""
SignSpeak - Training Pipeline
Training script for sign language recognition model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import os

from model import SignRecognitionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignLanguageDataset(Dataset):
    """
    Dataset for sign language recognition.
    Loads preprocessed landmark sequences.
    """
    
    def __init__(
        self,
        data_dir: Path,
        language: str = 'isl',
        split: str = 'train',
        max_seq_len: int = 64
    ):
        """
        Args:
            data_dir: Directory containing preprocessed data
            language: 'isl' or 'asl'
            split: 'train', 'val', or 'test'
            max_seq_len: Maximum sequence length (for padding/truncating)
        """
        self.data_dir = Path(data_dir)
        self.language = language
        self.split = split
        self.max_seq_len = max_seq_len
        
        # Load metadata
        metadata_path = self.data_dir / f"{language}_{split}_metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        logger.info(f"Loaded {len(self.metadata)} samples for {language} {split}")
        
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dict with keys:
                - landmarks: (seq_len, 543) tensor
                - label: int class label
                - padding_mask: (seq_len,) bool tensor
                - seq_len: actual sequence length
        """
        item = self.metadata[idx]
        
        # Load landmarks
        landmark_path = self.data_dir / item['landmark_file']
        landmarks = np.load(landmark_path)  # (original_seq_len, 543)
        
        actual_len = len(landmarks)
        
        # Pad or truncate to max_seq_len
        if actual_len < self.max_seq_len:
            # Pad with zeros
            padding = np.zeros((self.max_seq_len - actual_len, 543))
            landmarks = np.concatenate([landmarks, padding], axis=0)
            padding_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
            padding_mask[actual_len:] = True
        else:
            # Truncate
            landmarks = landmarks[:self.max_seq_len]
            padding_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
            actual_len = self.max_seq_len
        
        return {
            'landmarks': torch.from_numpy(landmarks).float(),
            'label': torch.tensor(item['label'], dtype=torch.long),
            'padding_mask': padding_mask,
            'seq_len': actual_len,
            'gloss': item.get('gloss', ''),
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch, num_classes) logits
            targets: (batch,) class labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


class Trainer:
    """Training pipeline for sign recognition model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        language: str = 'isl',
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        num_epochs: int = 100,
        save_dir: Path = Path('./checkpoints')
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.language = language
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
        # Optimizer with AdamW
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,
            eta_min=1e-6
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Metrics tracking
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
        
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            landmarks = batch['landmarks'].to(self.device)
            labels = batch['label'].to(self.device)
            padding_mask = batch['padding_mask'].to(self.device)
            
            # Forward pass with mixed precision
            with autocast():
                logits, _ = self.model(landmarks, self.language, padding_mask)
                loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(self.val_loader, desc='Validating'):
            landmarks = batch['landmarks'].to(self.device)
            labels = batch['label'].to(self.device)
            padding_mask = batch['padding_mask'].to(self.device)
            
            with autocast():
                logits, _ = self.model(landmarks, self.language, padding_mask)
                loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
        }
        
        # Save regular checkpoint
        path = self.save_dir / f'{self.language}_epoch_{epoch}.pth'
        torch.save(checkpoint, path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / f'{self.language}_best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f'Saved best model with accuracy: {val_acc:.4f}')
    
    def train(self):
        """Full training loop."""
        logger.info(f"Starting training for {self.language.upper()}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.num_epochs}")
        
        for epoch in range(1, self.num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{self.num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val Accuracy: {val_acc:.4f}")
            logger.info(f"Learning Rate: {current_lr:.6f}")
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, val_acc, is_best)
        
        logger.info(f"\nTraining completed!")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")


# Main training script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with processed data')
    parser.add_argument('--language', type=str, default='isl', choices=['isl', 'asl'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    args = parser.parse_args()
    
    # Create datasets
    train_dataset = SignLanguageDataset(
        args.data_dir,
        language=args.language,
        split='train'
    )
    
    val_dataset = SignLanguageDataset(
        args.data_dir,
        language=args.language,
        split='val'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    num_classes = 263 if args.language == 'isl' else 2000
    model = SignRecognitionModel(
        num_isl_classes=263,
        num_asl_classes=2000
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        language=args.language,
        device=args.device,
        learning_rate=args.lr,
        num_epochs=args.num_epochs,
        save_dir=Path(args.save_dir)
    )
    
    # Start training
    trainer.train()
