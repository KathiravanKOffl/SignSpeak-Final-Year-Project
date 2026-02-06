"""
SignSpeak - Kaggle Inference Server
====================================
This notebook runs a FastAPI server on Kaggle to serve sign language predictions.

SETUP INSTRUCTIONS:
1. Create NEW Kaggle notebook (Interactive mode)
2. Settings:
   - Type: Interactive Notebook
   - Accelerator: CPU (GPU optional)
   - Internet: ON
3. Add your model:
   - Click "Add Input" → "Models"
   - Search: "kathiravankoffl/signspeak-model"
   - Add it (private model)
4. Copy this code into cells
5. Run all cells
6. Copy the tunnel URL from output
7. Paste URL in frontend settings

"""

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================
print("📦 Installing dependencies...")
!pip install -q fastapi uvicorn[standard] nest-asyncio pydantic
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64
print("✓ Installation complete")

# ============================================================================
# CELL 2: Load Model from Kaggle Models
# ============================================================================
import torch
import json
import os

print("\n📂 Loading model from Kaggle Models...")

# Path to your Kaggle Model (adjust version number if needed)
MODEL_PATH = "/kaggle/input/signspeak-model/pytorch/default/1"

# Load the model package
model_file = os.path.join(MODEL_PATH, "sign_model.pth")
vocab_file = os.path.join(MODEL_PATH, "vocabulary.json")

print(f"Loading from: {MODEL_PATH}")

# Load model dictionary
model_dict = torch.load(model_file, map_location='cpu', weights_only=False)

# Extract parameters
input_size = model_dict['input_size']
hidden_size = model_dict['hidden_size']
num_layers = model_dict['num_layers']
num_classes = model_dict['num_classes']
vocabulary = model_dict['vocabulary']

print(f"✓ Model loaded")
print(f"  - Input size: {input_size}")
print(f"  - Vocabulary: {vocabulary}")
print(f"  - Classes: {num_classes}")

# Reconstruct model architecture
import torch.nn as nn

class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(SignLanguageLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

# Initialize model and load weights
model = SignLanguageLSTM(input_size, hidden_size, num_layers, num_classes)
model.load_state_dict(model_dict['model_state_dict'])
model.eval()

print("✓ Model ready for inference")

# ============================================================================
# CELL 3: Define Inference Function
# ============================================================================
import numpy as np

def predict_sign(landmarks_sequence):
    """
    Perform inference on a sequence of landmarks.
    
    Args:
        landmarks_sequence: List of 32 frames, each with 399 features
    
    Returns:
        dict with 'word' and 'confidence'
    """
    try:
        # Convert to tensor [1, 32, 399]
        input_tensor = torch.FloatTensor(landmarks_sequence).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
        
        predicted_word = vocabulary[predicted_idx]
        
        return {
            "word": predicted_word,
            "confidence": float(confidence),
            "status": "success"
        }
    
    except Exception as e:
        return {
            "word": None,
            "confidence": 0.0,
            "status": "error",
            "error": str(e)
        }

print("✓ Inference function ready")

# ============================================================================
# CELL 4: Create FastAPI Server
# ============================================================================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import nest_asyncio

# Enable nested async (required for Jupyter)
nest_asyncio.apply()

app = FastAPI(
    title="SignSpeak Inference API",
    description="Real-time sign language recognition",
    version="1.0.0"
)

# Enable CORS for all origins (frontend can access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class LandmarksInput(BaseModel):
    sequence: List[List[float]]  # [32, 399]

# API Routes
@app.get("/")
async def root():
    """Health check and model info"""
    return {
        "status": "online",
        "model": "SignSpeak LSTM",
        "vocabulary": vocabulary,
        "vocab_size": len(vocabulary),
        "input_shape": [32, input_size],
        "version": "1.0"
    }

@app.post("/predict")
async def predict(data: LandmarksInput):
    """Predict sign from landmarks sequence"""
    try:
        # Validate input shape
        if len(data.sequence) != 32:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 32 frames, got {len(data.sequence)}"
            )
        
        if len(data.sequence[0]) != input_size:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {input_size} features per frame, got {len(data.sequence[0])}"
            )
        
        # Run inference
        result = predict_sign(data.sequence)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vocabulary_loaded": len(vocabulary) > 0
    }

print("✓ FastAPI server configured")

# ============================================================================
# CELL 5: Start Server + Cloudflared Tunnel
# ============================================================================
import threading
import time

print("\n" + "=" * 70)
print("🚀 Starting SignSpeak Inference Server")
print("=" * 70)

# Start FastAPI server in background thread
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

time.sleep(5)
print("✓ FastAPI server started on port 8000")

# Start cloudflared tunnel
print("\n🌐 Starting cloudflared tunnel...")
print("=" * 70)
print("\n⚠️  COPY THE URL BELOW (https://xxx.trycloudflare.com)")
print("=" * 70 + "\n")

!./cloudflared-linux-amd64 tunnel --url http://localhost:8000
