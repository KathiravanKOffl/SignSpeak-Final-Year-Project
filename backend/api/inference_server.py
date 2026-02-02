"""
SignSpeak - FastAPI Inference Server
Serves sign language recognition predictions via REST API
Designed to run on Google Colab with Cloudflare Tunnel
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path

from model import SignRecognitionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SignSpeak Inference API",
    description="Real-time sign language recognition for ISL and ASL",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cloudflare Pages domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model: Optional[SignRecognitionModel] = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class name mappings
isl_classes: Dict[int, str] = {}
asl_classes: Dict[int, str] = {}


class LandmarkRequest(BaseModel):
    """Request schema for landmark-based inference."""
    landmarks: List[List[float]]  # (seq_len, 543) flattened landmarks
    language: str = 'isl'  # 'isl' or 'asl'
    top_k: int = 5  # Return top-k predictions


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    predictions: List[Dict[str, Any]]  # List of {class: str, confidence: float}
    gloss: str  # Most likely sign
    confidence: float
    language: str
    processing_time_ms: float


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model, isl_classes, asl_classes
    
    logger.info(f"Loading model on device: {device}")
    
    try:
        # Create model with ISL-123 configuration
        model = SignRecognitionModel(
            input_dim=408,  # 136 landmarks × 3
            hidden_dim=384,  # Match trained model
            num_isl_classes=123,
            num_asl_classes=123,  # Placeholder
            num_heads=8,
            num_layers=4,
            dropout=0.4
        ).to(device)
        
        # Load weights for ISL-123 model
        checkpoint_path = Path('./checkpoints/best_isl_123.pth')
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"✅ Loaded ISL-123 checkpoint from {checkpoint_path}")
        else:
            logger.warning(f"⚠️ No checkpoint found at {checkpoint_path}, using random weights")
        
        model.eval()
        
        # Load class mappings from JSON
        mapping_path = Path('./checkpoints/label_mapping_123.json')
        if mapping_path.exists():
            import json
            with open(mapping_path, 'r') as f:
                mapping_data = json.load(f)
                # Expect format: {"0": "adult", "1": "alright", ...}
                isl_classes = {int(k): v for k, v in mapping_data.items()}
            logger.info(f"✅ Loaded {len(isl_classes)} ISL class mappings")
        else:
            # Fallback to dummy mappings
            logger.warning("⚠️ No label mapping found, using placeholders")
            isl_classes = {i: f"ISL_SIGN_{i}" for i in range(123)}
        
        # ASL not implemented yet
        asl_classes = {i: f"ASL_SIGN_{i}" for i in range(123)}
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "model_loaded": model is not None,
        "device": str(device),
        "supported_languages": ["isl", "asl"]
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model": {
            "loaded": model is not None,
            "device": str(device),
            "isl_classes": len(isl_classes),
            "asl_classes": len(asl_classes),
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: LandmarkRequest):
    """
    Predict sign from landmark sequence.
    
    Args:
        request: LandmarkRequest with landmarks and language
        
    Returns:
        PredictionResponse with top-k predictions
    """
    import time
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate language
        if request.language.lower() not in ['isl', 'asl']:
            raise HTTPException(status_code=400, detail="Language must be 'isl' or 'asl'")
        
        # Convert landmarks to tensor
        landmarks_array = np.array(request.landmarks, dtype=np.float32)
        
        # Reshape if needed
        if len(landmarks_array.shape) == 1:
            # Assuming flattened (seq_len * 543)
            seq_len = len(landmarks_array) // 543
            landmarks_array = landmarks_array.reshape(seq_len, 543)
        
        landmarks_tensor = torch.from_numpy(landmarks_array).unsqueeze(0).to(device)  # (1, seq_len, 543)
        
        # Inference
        with torch.no_grad():
            logits, features = model(landmarks_tensor, language=request.language.lower())
            probabilities = torch.softmax(logits, dim=1)[0]  # (num_classes,)
        
        # Get top-k predictions
        top_k_probs, top_k_indices = torch.topk(probabilities, k=min(request.top_k, len(probabilities)))
        
        # Select class mapping
        class_map = isl_classes if request.language.lower() == 'isl' else asl_classes
        
        # Format predictions
        predictions = [
            {
                "class": class_map.get(idx.item(), f"UNKNOWN_{idx.item()}"),
                "confidence": prob.item()
            }
            for prob, idx in zip(top_k_probs, top_k_indices)
        ]
        
        # Processing time
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            predictions=predictions,
            gloss=predictions[0]["class"],
            confidence=predictions[0]["confidence"],
            language=request.language.lower(),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
async def batch_predict(requests: List[LandmarkRequest]):
    """
    Batch prediction for multiple sequences.
    
    Args:
        requests: List of LandmarkRequests
        
    Returns:
        List of PredictionResponses
    """
    results = []
    for req in requests:
        try:
            result = await predict(req)
            results.append(result)
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            results.append({"error": str(e)})
    
    return results


# Run server
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting SignSpeak Inference Server")
    logger.info(f"Device: {device}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
