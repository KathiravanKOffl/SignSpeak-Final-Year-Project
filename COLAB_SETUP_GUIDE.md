# 🚀 Colab Inference Server - Copy-Paste Guide

Follow these steps to run your ASL model on Google Colab with Cloudflare Tunnel.

**No account needed!** Just copy-paste each cell below.

---

## Setup

1. Go to https://colab.research.google.com/
2. Create new notebook
3. Change runtime: **Runtime → Change runtime type → GPU (T4)**
4. Copy-paste each code block below into cells

---

## Cell 1: Install Dependencies

```python
!pip install -q fastapi uvicorn torch mediapipe numpy pydantic nest-asyncio
print("✅ Dependencies installed!")
```

---

## Cell 2: Install Cloudflare Tunnel

```python
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64
!mv cloudflared-linux-amd64 /usr/local/bin/cloudflared
print("✅ Cloudflare Tunnel installed!")
```

---

## Cell 3: Upload Model Files

```python
from google.colab import files
import os

os.makedirs('/content/checkpoints', exist_ok=True)

print("📤 Upload best_model.pth and label_mapping.json:")
uploaded = files.upload()

for filename in uploaded.keys():
    os.rename(filename, f'/content/checkpoints/{filename}')
    print(f"✅ {filename}")

!ls -lh /content/checkpoints/
```

**Click "Choose Files" and upload:**
- `best_model.pth` (from Downloads)
- `label_mapping.json` (from Downloads)

---

## Cell 4: Define Model

```python
import torch
import torch.nn as nn

class TemporalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, dropout):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        forward_h = h_n[-2]
        backward_h = h_n[-1]
        combined = torch.cat([forward_h, backward_h], dim=1)
        return self.classifier(combined)

print("✅ Model architecture defined")
```

---

## Cell 5: Load Model

```python
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load checkpoint
checkpoint = torch.load('/content/checkpoints/best_model.pth', map_location=device)

config = checkpoint.get('config', {})
num_classes = checkpoint.get('num_classes', 25)
input_dim = checkpoint.get('input_dim', 150)
hidden_dim = config.get('hidden_dim', 256)
num_layers = config.get('lstm_layers', 2)
dropout = config.get('dropout', 0.4)

# Create model
model = TemporalLSTM(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_classes=num_classes,
    num_layers=num_layers,
    dropout=dropout
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

# Load labels
with open('/content/checkpoints/label_mapping.json', 'r') as f:
    label_data = json.load(f)
    id_to_label = {int(k): v for k, v in label_data['id_to_label'].items()}

print(f"✅ {len(id_to_label)} classes: {list(id_to_label.values())}")
```

---

## Cell 6: Create API Server

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import numpy as np
import time

app = FastAPI(title="SignSpeak ASL Inference")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LandmarkRequest(BaseModel):
    landmarks: Dict[str, Any]
    language: str = 'asl'
    top_k: int = 5

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "SignSpeak ASL",
        "model": "TemporalLSTM",
        "device": str(device)
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "gpu": torch.cuda.is_available()
    }

@app.post("/predict")
def predict(req: LandmarkRequest):
    start = time.time()
    try:
        lm = req.landmarks
        
        # Extract right hand landmarks
        right_hand = np.array(lm.get('rightHand', []))
        if len(right_hand) == 0:
            return {"error": "No hand detected"}
        
        # Flatten and pad to 150
        hand_flat = right_hand.flatten()[:150]
        if len(hand_flat) < 150:
            hand_flat = np.pad(hand_flat, (0, 150 - len(hand_flat)))
        
        # Repeat for 30 frames
        input_tensor = torch.FloatTensor(
            np.tile(hand_flat, (30, 1))
        ).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            top_probs, top_idx = torch.topk(probs, min(req.top_k, num_classes))
        
        predictions = [
            {
                "gloss": id_to_label[top_idx[0][i].item()],
                "confidence": float(top_probs[0][i])
            }
            for i in range(len(top_idx[0]))
        ]
        
        return {
            "predictions": predictions,
            "gloss": predictions[0]["gloss"],
            "confidence": predictions[0]["confidence"],
            "processing_time_ms": (time.time() - start) * 1000
        }
    except Exception as e:
        return {"error": str(e)}

print("✅ FastAPI server created")
```

---

## Cell 7: Start Server + Tunnel

```python
import nest_asyncio
import uvicorn
from threading import Thread
import subprocess
import time
import re

nest_asyncio.apply()

# Start FastAPI
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

Thread(target=run_server, daemon=True).start()
print("⏳ Starting server...")
time.sleep(3)
print("✅ Server running on port 8000\n")

# Start Cloudflare Tunnel
print("🌐 Starting Cloudflare Tunnel...")
tunnel = subprocess.Popen(
    ['cloudflared', 'tunnel', '--url', 'http://localhost:8000'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Extract URL
tunnel_url = None
for line in tunnel.stderr:
    if 'trycloudflare.com' in line:
        match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
        if match:
            tunnel_url = match.group(0)
            break

if tunnel_url:
    print("\n" + "="*70)
    print("🎉 SERVER IS LIVE!")
    print("="*70)
    print(f"\n📡 URL: {tunnel_url}")
    print(f"\n🔧 Add to Cloudflare Pages env vars:")
    print(f"   COLAB_TUNNEL_URL = {tunnel_url}")
    print(f"\n✅ Test: !curl {tunnel_url}/health")
    print("\n⚠️  Keep this cell running!")
    print("="*70)
else:
    print("❌ Could not get tunnel URL")

# Keep alive
try:
    while True:
        time.sleep(60)
        print(f"✅ {time.strftime('%H:%M:%S')} - {tunnel_url}")
except KeyboardInterrupt:
    tunnel.terminate()
```

**This cell will keep running - don't stop it!**

---

## Done! 🎉

**Copy the tunnel URL** (looks like `https://xxx.trycloudflare.com`)

**Add to Cloudflare:**
1. Cloudflare Pages → Your Project → Settings → Environment Variables
2. Add: `COLAB_TUNNEL_URL = https://xxx.trycloudflare.com`
3. Save

**Deploy:**
```bash
git add .
git commit -m "feat: complete ASL system"
git push origin main
```

---

## Daily Restart

Tunnel URL changes daily. Just re-run **Cell 7** each morning:
1. Stop the cell (⏹️)
2. Run it again (▶️)
3. Copy new URL
4. Update Cloudflare env var
