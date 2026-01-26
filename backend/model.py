"""
SignSpeak - Sign Language Recognition Model
Hybrid CNN-Transformer Architecture for ISL and ASL Recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SpatialFeatureExtractor(nn.Module):
    """
    CNN-based spatial feature extraction from landmark sequences.
    Processes normalized landmark coordinates to extract spatial patterns.
    """
    
    def __init__(self, input_dim: int = 543, hidden_dim: int = 256):
        """
        Args:
            input_dim: Total number of landmark features (543 for pose+hands+face)
            hidden_dim: Dimension of hidden features
        """
        super().__init__()
        
        # Landmark embedding: Project raw coordinates to higher dimension
        self.landmark_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Temporal convolutions to capture motion patterns
        self.conv_layers = nn.ModuleList([
            # Conv1: Capture frame-level features
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            # Conv2: Capture short-term motion
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            # Conv3: Capture medium-term motion
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(3)
        ])
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) landmark sequences
            
        Returns:
            features: (batch, seq_len, hidden_dim) spatial features
        """
        # Embed landmarks
        x = self.landmark_embed(x)  # (batch, seq_len, hidden_dim)
        
        # Transpose for Conv1d: (batch, hidden_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Multi-scale temporal convolutions
        conv_outputs = []
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            out = conv(x)
            out = norm(out)
            out = F.relu(out)
            out = self.dropout(out)
            conv_outputs.append(out)
        
        # Concatenate multi-scale features
        x = torch.stack(conv_outputs, dim=0).mean(dim=0)  # Average pooling
        
        # Transpose back: (batch, seq_len, hidden_dim)
        x = x.transpose(1, 2)
        
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for temporal modeling of sign language gestures.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: (batch, seq_len, d_model) input features
            src_mask: Optional attention mask
            src_key_padding_mask: (batch, seq_len) bool mask for padding
            
        Returns:
            output: (batch, seq_len, d_model) encoded features
        """
        src = self.pos_encoder(src)
        src = self.dropout(src)
        
        output = self.transformer(
            src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        return output


class SignRecognitionModel(nn.Module):
    """
    Hybrid CNN-Transformer model for sign language recognition.
    Supports both ISL (263 classes) and ASL (2000 classes).
    """
    
    def __init__(
        self,
        input_dim: int = 543,  # pose(33*4) + hands(21*3*2) + face(50*3)
        hidden_dim: int = 256,
        num_isl_classes: int = 263,
        num_asl_classes: int = 2000,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Spatial feature extractor (CNN)
        self.spatial_extractor = SpatialFeatureExtractor(input_dim, hidden_dim)
        
        # Temporal encoder (Transformer)
        self.temporal_encoder = TransformerEncoder(
            d_model=hidden_dim,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification heads
        self.isl_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_isl_classes)
        )
        
        self.asl_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_asl_classes)
        )
        
    def forward(
        self,
        landmarks: torch.Tensor,
        language: str = 'isl',
        padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            landmarks: (batch, seq_len, input_dim) normalized landmark sequences
            language: 'isl' or 'asl' for selecting classification head
            padding_mask: (batch, seq_len) bool mask for padding frames
            
        Returns:
            logits: (batch, num_classes) classification logits
            features: (batch, hidden_dim) global features for embeddings
        """
        # Extract spatial features
        features = self.spatial_extractor(landmarks)  # (batch, seq_len, hidden_dim)
        
        # Encode temporal patterns
        encoded = self.temporal_encoder(
            features,
            src_key_padding_mask=padding_mask
        )  # (batch, seq_len, hidden_dim)
        
        # Global pooling
        pooled = encoded.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        pooled = self.global_pool(pooled).squeeze(-1)  # (batch, hidden_dim)
        
        # Classification
        if language.lower() == 'isl':
            logits = self.isl_classifier(pooled)
        elif language.lower() == 'asl':
            logits = self.asl_classifier(pooled)
        else:
            raise ValueError(f"Unknown language: {language}. Must be 'isl' or 'asl'")
        
        return logits, pooled


# Helper function to count parameters
def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage
if __name__ == "__main__":
    # Create model
    model = SignRecognitionModel(
        input_dim=543,
        hidden_dim=256,
        num_isl_classes=263,
        num_asl_classes=2000,
        nhead=8,
        num_layers=6
    )
    
    print(f"Model created with {count_parameters(model):,} trainable parameters")
    
    # Test forward pass
    batch_size = 4
    seq_len = 64  # 64 frames (~2 seconds at 30 fps)
    
    # Dummy input
    landmarks = torch.randn(batch_size, seq_len, 543)
    
    # ISL inference
    logits_isl, features = model(landmarks, language='isl')
    print(f"\nISL output shape: {logits_isl.shape}")  # (4, 263)
    print(f"Features shape: {features.shape}")  # (4, 256)
    
    # ASL inference
    logits_asl, _ = model(landmarks, language='asl')
    print(f"ASL output shape: {logits_asl.shape}")  # (4, 2000)
    
    # Test with padding mask
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask[:, 50:] = True  # Mask frames after 50
    
    logits_masked, _ = model(landmarks, language='isl', padding_mask=padding_mask)
    print(f"\nMasked output shape: {logits_masked.shape}")  # (4, 263)
