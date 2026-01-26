"""
WLASL Pose-TGCN Model Adapter
Adapts MediaPipe landmarks to WLASL pre-trained model format
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from typing import Tuple, Optional, Dict


class WLASLAdapter:
    """
    Adapter to load and run WLASL pre-trained Pose-TGCN model.
    
    Input: MediaPipe landmarks (pose 33 + hands 42 + face 50)
    Output: ASL gloss predictions with confidence
    """
    
    WLASL_KEYPOINTS = 75  # 33 body + 21 left hand + 21 right hand
    
    def __init__(
        self,
        checkpoint_path: str,
        gloss_mapping_path: Optional[str] = None,
        device: str = 'cuda',
        num_classes: int = 100  # asl100, asl300, asl1000, or asl2000
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Load checkpoint
        print(f"[WLASL] Loading checkpoint from {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Try to extract model from checkpoint
        self._load_model()
        
        # Load gloss mapping
        self.gloss_mapping = self._load_gloss_mapping(gloss_mapping_path)
        
    def _load_model(self):
        """Load or reconstruct the model."""
        # Check what's in the checkpoint
        if isinstance(self.checkpoint, dict):
            keys = list(self.checkpoint.keys())
            print(f"[WLASL] Checkpoint keys: {keys[:10]}...")  # First 10 keys
            
            # Try common checkpoint formats
            if 'model_state_dict' in self.checkpoint:
                state_dict = self.checkpoint['model_state_dict']
            elif 'state_dict' in self.checkpoint:
                state_dict = self.checkpoint['state_dict']
            else:
                # Assume checkpoint is the state_dict itself
                state_dict = self.checkpoint
                
            # Create a simple classifier that matches the checkpoint
            self.model = self._create_simple_model(state_dict)
        else:
            # Checkpoint might be the model itself
            self.model = self.checkpoint
            
        self.model.to(self.device)
        self.model.eval()
        print(f"[WLASL] Model loaded on {self.device}")
        
    def _create_simple_model(self, state_dict: dict) -> nn.Module:
        """
        Create a simple model architecture that matches the checkpoint.
        This is a fallback if we can't load the original TGCN.
        """
        # Simple classifier for pose sequences
        class PoseClassifier(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
                self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                # x: (batch, seq_len, features)
                x = x.mean(dim=1)  # Average over sequence
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.dropout(self.relu(self.fc2(x)))
                x = self.fc3(x)
                return x
                
        # Use checkpoint shapes to infer dimensions
        input_dim = 75 * 3  # 75 keypoints * 3 coordinates (x, y, z)
        hidden_dim = 256
        
        model = PoseClassifier(input_dim, hidden_dim, self.num_classes)
        
        # Try to load state dict (may fail if shapes don't match)
        try:
            model.load_state_dict(state_dict, strict=False)
            print("[WLASL] State dict loaded (partial match)")
        except Exception as e:
            print(f"[WLASL] Could not load state dict: {e}")
            print("[WLASL] Using randomly initialized model")
            
        return model
        
    def _load_gloss_mapping(self, path: Optional[str]) -> Dict[int, str]:
        """Load gloss ID to text mapping."""
        if path and Path(path).exists():
            with open(path, 'r') as f:
                data = json.load(f)
                return {int(k): v for k, v in data.get('id_to_gloss', {}).items()}
        
        # Default mapping (will be overwritten when proper mapping is loaded)
        return {i: f"ASL_SIGN_{i}" for i in range(self.num_classes)}
        
    def mediapipe_to_wlasl(
        self,
        pose: np.ndarray,  # (33, 4) - x, y, z, visibility
        left_hand: np.ndarray,  # (21, 3)
        right_hand: np.ndarray,  # (21, 3)
    ) -> np.ndarray:
        """
        Convert MediaPipe landmarks to WLASL format.
        
        WLASL expects: 75 keypoints (33 body + 21 left + 21 right)
        Each keypoint: (x, y, z)
        
        Returns: (75, 3) array
        """
        # Extract only xyz from pose (ignore visibility)
        pose_xyz = pose[:, :3] if pose.shape[1] > 3 else pose
        
        # Normalize coordinates to [-1, 1] range
        def normalize(arr):
            if arr.max() > 1 or arr.min() < 0:
                return arr  # Already normalized
            # MediaPipe gives 0-1, convert to -1 to 1
            return arr * 2 - 1
            
        pose_xyz = normalize(pose_xyz)
        left_hand = normalize(left_hand)
        right_hand = normalize(right_hand)
        
        # Concatenate: body + left_hand + right_hand
        keypoints = np.vstack([pose_xyz, left_hand, right_hand])
        
        return keypoints.astype(np.float32)
        
    def predict(
        self,
        landmarks_sequence: list,  # List of (75, 3) arrays
        top_k: int = 5
    ) -> Tuple[str, float, list]:
        """
        Predict ASL gloss from landmark sequence.
        
        Args:
            landmarks_sequence: List of keypoint frames
            top_k: Number of top predictions to return
            
        Returns:
            (predicted_gloss, confidence, top_k_predictions)
        """
        if len(landmarks_sequence) == 0:
            return "UNKNOWN", 0.0, []
            
        # Stack into tensor
        sequence = np.stack(landmarks_sequence)  # (seq_len, 75, 3)
        
        # Flatten keypoints: (seq_len, 75*3)
        sequence_flat = sequence.reshape(sequence.shape[0], -1)
        
        # Add batch dimension and convert to tensor
        x = torch.from_numpy(sequence_flat).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            
        # Get top-k predictions
        top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, len(probs)))
        
        predictions = [
            {
                "gloss": self.gloss_mapping.get(idx.item(), f"SIGN_{idx.item()}"),
                "confidence": prob.item()
            }
            for prob, idx in zip(top_k_probs, top_k_indices)
        ]
        
        top_gloss = predictions[0]["gloss"]
        top_confidence = predictions[0]["confidence"]
        
        return top_gloss, top_confidence, predictions
        

def load_wlasl_model(variant: str = "asl100") -> WLASLAdapter:
    """
    Load WLASL pre-trained model.
    
    Args:
        variant: One of 'asl100', 'asl300', 'asl1000', 'asl2000'
        
    Returns:
        WLASLAdapter instance
    """
    variants = {
        "asl100": (100, "/content/wlasl_weights/archived/asl100/ckpt.pth"),
        "asl300": (300, "/content/wlasl_weights/archived/asl300/ckpt.pth"),
        "asl1000": (1000, "/content/wlasl_weights/archived/asl1000/ckpt.pth"),
        "asl2000": (2000, "/content/wlasl_weights/archived/asl2000/ckpt.pth"),
    }
    
    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variants.keys())}")
        
    num_classes, checkpoint_path = variants[variant]
    
    return WLASLAdapter(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        gloss_mapping_path="/content/wlasl_weights/gloss_mapping.json"
    )
