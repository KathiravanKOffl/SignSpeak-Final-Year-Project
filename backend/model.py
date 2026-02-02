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
        
        # Input Embedding (name must match checkpoint: 'embed')
        self.embed = nn.Sequential(
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
        
        # Classification Head (name must match checkpoint: 'classifier')
        # Note: Checkpoint only has one classifier, used for ISL
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_isl_classes)
        )
        
    def forward(self, x, language='isl'):
        """
        Args:
            x: Input landmarks (batch_size, seq_len, input_dim)
            language: 'isl' or 'asl'
        Returns:
            logits: Prediction scores
            features: Extracted temporal features
        """
        # Embed input
        x = self.embed(x)  # (batch, seq_len, hidden_dim)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch, seq_len, hidden_dim)
        
        # Pool: mean over sequence
        x = x.mean(dim=1)  # (batch, hidden_dim)
        
        # Classify
        logits = self.classifier(x)  # (batch, num_classes)
        return logits
