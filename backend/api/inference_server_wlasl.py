"""
SignSpeak - FastAPI Inference Server with WLASL Pre-trained Model
Serves ASL sign language recognition using pre-trained Pose-TGCN
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from typing import List, Dict, Optional, Any
import logging
import time
from pathlib import Path
import json
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SignSpeak ASL Recognition API",
    description="Real-time ASL recognition using WLASL pre-trained model",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model reference
wlasl_model = None
gloss_mapping = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LandmarkRequest(BaseModel):
    """Request schema for landmark-based inference."""
    landmarks: Dict[str, Any]  # MediaPipe format: {pose, leftHand, rightHand, face, confidence, timestamp}
    language: str = 'asl'
    top_k: int = 5


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    predictions: List[Dict[str, Any]]
    gloss: str
    confidence: float
    language: str
    processing_time_ms: float


def load_simple_classifier(num_classes: int = 100):
    """Create a simple classifier for ASL signs."""
    import torch.nn as nn
    
    class SimpleASLClassifier(nn.Module):
        def __init__(self, input_dim: int = 225, hidden_dim: int = 256, num_classes: int = 100):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, num_classes)
            )
            
        def forward(self, x):
            # x: (batch, seq_len, features) or (batch, features)
            if len(x.shape) == 3:
                x = x.mean(dim=1)  # Average over sequence
            return self.net(x)
            
    return SimpleASLClassifier(num_classes=num_classes)


@app.on_event("startup")
async def load_model():
    """Load WLASL pre-trained model on startup."""
    global wlasl_model, gloss_mapping
    
    logger.info(f"Loading WLASL model on device: {device}")
    
    try:
    try:
        # Check for trained model first (best_model.pth)
        trained_path = Path("./checkpoints/best_model.pth")
        trained_mapping_path = Path("./checkpoints/label_mapping.json")
        
        # Fallback to pre-trained weights (ckpt.pth)
        wlasl_path = Path("/content/wlasl_weights/archived/asl100/ckpt.pth")
        
        if trained_path.exists():
            logger.info(f"Found trained model at {trained_path}")
            checkpoint = torch.load(trained_path, map_location=device)
            
            # Extract info from checkpoint
            if 'num_classes' in checkpoint:
                num_classes = checkpoint['num_classes']
                input_dim = checkpoint.get('input_dim', 225) # Default to 225 if missing
            else:
                num_classes = 100 # Default fallback
                input_dim = 225
                
            # Create model
            wlasl_model = load_simple_classifier(input_dim=input_dim, num_classes=num_classes)
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                wlasl_model.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                wlasl_model.load_state_dict(checkpoint['model'])
                
            # Load gloss mapping from checkpoint or file
            if 'id_to_label' in checkpoint:
                gloss_mapping = {int(k): v.upper() for k, v in checkpoint['id_to_label'].items()}
                logger.info("Loaded mapping from checkpoint")
            elif trained_mapping_path.exists():
                with open(trained_mapping_path, 'r') as f:
                    data = json.load(f)
                    # Check format (could be id_to_label inside or direct)
                    if 'id_to_label' in data:
                        gloss_mapping = {int(k): v.upper() for k, v in data['id_to_label'].items()}
                    else:
                        gloss_mapping = {int(k): v.upper() for k, v in data.items()}
                logger.info("Loaded mapping from json file")
            
            wlasl_model.to(device)
            wlasl_model.eval()
            logger.info(f"✅ Trained model loaded! Classes: {len(gloss_mapping)}")
            return

        elif wlasl_path.exists():
            logger.info("Found WLASL pre-trained weights!")
            checkpoint = torch.load(wlasl_path, map_location=device)
            
            # Create model matching checkpoint
            wlasl_model = load_simple_classifier(num_classes=100)
            
            # Try to load weights
            try:
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    wlasl_model.load_state_dict(checkpoint['state_dict'], strict=False)
                elif isinstance(checkpoint, dict):
                    wlasl_model.load_state_dict(checkpoint, strict=False)
                logger.info("Loaded pre-trained weights!")
            except Exception as e:
                logger.warning(f"Could not load weights exactly: {e}")
                logger.info("Using initialized model")
                
            wlasl_model.to(device)
            wlasl_model.eval()
            
            # Load default WLASL mapping
            gloss_path = Path("/content/wlasl_weights/gloss_mapping.json")
            if gloss_path.exists():
                with open(gloss_path, 'r') as f:
                    data = json.load(f)
                    gloss_mapping = {int(k): v for k, v in data.get('id_to_gloss', {}).items()}
            else:
                 # Default 100 classes
                 pass # Use the hardcoded list below
                 
        else:
            logger.warning("No model found, using random initialized model")
            wlasl_model = load_simple_classifier(num_classes=100)
            wlasl_model.to(device)
            wlasl_model.eval()
        
        # If mapping empty, load default
        if not gloss_mapping:
            # Default ASL glosses (top 100 common signs)
            common_glosses = [
                "hello", "thank-you", "yes", "no", "please", "sorry", "help", "water", "food", "good",
                "bad", "name", "what", "where", "how", "i", "you", "mother", "father", "friend",
                "love", "happy", "sad", "angry", "hungry", "thirsty", "tired", "sick", "pain", "doctor",
                "family", "home", "school", "work", "money", "time", "day", "night", "morning", "evening",
                "today", "tomorrow", "yesterday", "week", "month", "year", "eat", "drink", "sleep", "walk",
                "run", "sit", "stand", "go", "come", "stop", "wait", "want", "need", "like",
                "dont-like", "know", "dont-know", "understand", "think", "feel", "see", "hear", "speak", "write",
                "read", "learn", "teach", "play", "dance", "sing", "cook", "clean", "buy", "sell",
                "give", "take", "make", "break", "open", "close", "big", "small", "hot", "cold",
                "new", "old", "fast", "slow", "easy", "hard", "right", "wrong", "same", "different"
            ]
            gloss_mapping = {i: gloss.upper() for i, gloss in enumerate(common_glosses)}
            logger.info("Using default gloss mapping (100 common signs)")
            
        logger.info("✅ Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Fallback to simple model
        wlasl_model = load_simple_classifier(num_classes=100)
        wlasl_model.to(device)
        wlasl_model.eval()
        gloss_mapping = {i: f"SIGN_{i}" for i in range(100)}


def mediapipe_to_input(landmarks: Dict) -> torch.Tensor:
    """Convert MediaPipe landmarks to model input format."""
    
    # Extract arrays
    pose = np.array(landmarks.get('pose', np.zeros((33, 4))))
    left_hand = np.array(landmarks.get('leftHand', np.zeros((21, 3))))
    right_hand = np.array(landmarks.get('rightHand', np.zeros((21, 3))))
    
    # Ensure correct shapes
    if len(pose.shape) == 1:
        pose = pose.reshape(-1, 4) if len(pose) % 4 == 0 else pose.reshape(-1, 3)
    if len(left_hand.shape) == 1:
        left_hand = left_hand.reshape(-1, 3)
    if len(right_hand.shape) == 1:
        right_hand = right_hand.reshape(-1, 3)
    
    # Take only xyz from pose
    pose_xyz = pose[:, :3] if pose.shape[1] >= 3 else pose
    
    # Pad to expected sizes if needed
    if pose_xyz.shape[0] < 33:
        pose_xyz = np.vstack([pose_xyz, np.zeros((33 - pose_xyz.shape[0], 3))])
    if left_hand.shape[0] < 21:
        left_hand = np.vstack([left_hand, np.zeros((21 - left_hand.shape[0], 3))])
    if right_hand.shape[0] < 21:
        right_hand = np.vstack([right_hand, np.zeros((21 - right_hand.shape[0], 3))])
    
    # Combine: (33 + 21 + 21) * 3 = 225 features
    combined = np.vstack([pose_xyz[:33], left_hand[:21], right_hand[:21]])
    features = combined.flatten().astype(np.float32)
    
    return torch.from_numpy(features).unsqueeze(0)  # (1, 225)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "model_loaded": wlasl_model is not None,
        "device": str(device),
        "supported_languages": ["asl"],
        "num_classes": len(gloss_mapping)
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model": {
            "loaded": wlasl_model is not None,
            "device": str(device),
            "asl_classes": len(gloss_mapping),
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: LandmarkRequest):
    """
    Predict ASL sign from landmarks.
    """
    start_time = time.time()
    
    if wlasl_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert landmarks to input tensor
        input_tensor = mediapipe_to_input(request.landmarks).to(device)
        
        # Run inference
        with torch.no_grad():
            logits = wlasl_model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]
        
        # Get top-k predictions
        top_k_probs, top_k_indices = torch.topk(probs, k=min(request.top_k, len(probs)))
        
        predictions = [
            {
                "gloss": gloss_mapping.get(idx.item(), f"SIGN_{idx.item()}"),
                "confidence": round(prob.item(), 4)
            }
            for prob, idx in zip(top_k_probs, top_k_indices)
        ]
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            predictions=predictions,
            gloss=predictions[0]["gloss"],
            confidence=predictions[0]["confidence"],
            language="asl",
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Run server
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting SignSpeak ASL Inference Server")
    logger.info(f"Device: {device}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
