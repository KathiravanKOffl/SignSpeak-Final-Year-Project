import torch
import torch.nn as nn
import torch.nn.functional as F

class SignRecognitionModel(nn.Module):
    """
    CNN-Transformer Hybrid Model for Sign Language Recognition.
    Processes sequences of MediaPipe landmarks to predict signs.
    """
    def __init__(
        self, 
        input_dim: int = 408,  # 136 landmarks × 3 coords (updated for ISL-123)
        hidden_dim: int = 384,  # Match trained model
        num_isl_classes: int = 123,  # Actual number of ISL classes
        num_asl_classes: int = 123,  # Placeholder (not used yet)
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.4  # Match training config
    ):
        super().__init__()
        
        # Input Embedding
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification Heads
        self.isl_head = nn.Linear(hidden_dim, num_isl_classes)
        self.asl_head = nn.Linear(hidden_dim, num_asl_classes)
        
    def forward(self, x, language='isl'):
        """
        Args:
            x: Input landmarks (batch_size, seq_len, input_dim)
            language: 'isl' or 'asl'
        Returns:
            logits: Prediction scores
            features: Extracted temporal features
        """
        # Embed landmarks
        x = self.embedding(x) # (B, S, H)
        
        # Process sequence with Transformer
        x = self.transformer(x) # (B, S, H)
        
        # Global Average Pooling over temporal dimension
        features = x.mean(dim=1) # (B, H)
        
        # Select appropriate head
        if language.lower() == 'isl':
            logits = self.isl_head(features)
        else:
            logits = self.asl_head(features)
            
        return logits, features
