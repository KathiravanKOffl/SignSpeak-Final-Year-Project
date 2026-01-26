# SignSpeak Colab Setup - Complete Guide

Copy-paste each cell into Google Colab. Run in order.

---

## Cell 1: Setup Environment

```python
# Install dependencies
print("📦 Installing dependencies...")
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q fastapi uvicorn[standard] pydantic python-multipart
!pip install -q pycloudflared gdown

import torch
print(f"✅ PyTorch {torch.__version__}")
print(f"🖥️ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
```

---

## Cell 2: Clone Repository

```python
import os

# Clean slate
if os.path.exists('/content/SignSpeak-Final-Year-Project'):
    !rm -rf /content/SignSpeak-Final-Year-Project

# Clone
!git clone https://github.com/KathiravanKOffl/SignSpeak-Final-Year-Project.git
%cd /content/SignSpeak-Final-Year-Project/backend

print(f"✅ Cloned! Current dir: {os.getcwd()}")
```

---

## Cell 3: Download Pre-Trained WLASL Model

```python
import os
import gdown

print("📥 Downloading WLASL pre-trained model...")

os.makedirs('/content/wlasl_weights', exist_ok=True)

# Pose-TGCN weights (ASL 100/300/1000/2000)
gdown.download(
    'https://drive.google.com/uc?id=1dzvocsaylRsjqaY4r_lyRihPZn0I6AA_', 
    '/content/wlasl_weights/pose_tgcn.zip',
    quiet=False
)

!unzip -o /content/wlasl_weights/pose_tgcn.zip -d /content/wlasl_weights/

print("✅ Model downloaded!")
print("📂 Available:", os.listdir('/content/wlasl_weights/archived/'))
```

---

## Cell 4: Create Gloss Mapping

```python
import json

# Top 100 common ASL signs
common_glosses = [
    "HELLO", "THANK-YOU", "YES", "NO", "PLEASE", "SORRY", "HELP", "WATER", "FOOD", "GOOD",
    "BAD", "NAME", "WHAT", "WHERE", "HOW", "I", "YOU", "MOTHER", "FATHER", "FRIEND",
    "LOVE", "HAPPY", "SAD", "ANGRY", "HUNGRY", "THIRSTY", "TIRED", "SICK", "PAIN", "DOCTOR",
    "FAMILY", "HOME", "SCHOOL", "WORK", "MONEY", "TIME", "DAY", "NIGHT", "MORNING", "EVENING",
    "TODAY", "TOMORROW", "YESTERDAY", "WEEK", "MONTH", "YEAR", "EAT", "DRINK", "SLEEP", "WALK",
    "RUN", "SIT", "STAND", "GO", "COME", "STOP", "WAIT", "WANT", "NEED", "LIKE",
    "DONT-LIKE", "KNOW", "DONT-KNOW", "UNDERSTAND", "THINK", "FEEL", "SEE", "HEAR", "SPEAK", "WRITE",
    "READ", "LEARN", "TEACH", "PLAY", "DANCE", "SING", "COOK", "CLEAN", "BUY", "SELL",
    "GIVE", "TAKE", "MAKE", "BREAK", "OPEN", "CLOSE", "BIG", "SMALL", "HOT", "COLD",
    "NEW", "OLD", "FAST", "SLOW", "EASY", "HARD", "RIGHT", "WRONG", "SAME", "DIFFERENT"
]

gloss_mapping = {
    "gloss_to_id": {g: i for i, g in enumerate(common_glosses)},
    "id_to_gloss": {i: g for i, g in enumerate(common_glosses)}
}

with open('/content/wlasl_weights/gloss_mapping.json', 'w') as f:
    json.dump(gloss_mapping, f)

print(f"✅ Created mapping for {len(common_glosses)} signs")
print("Sample:", common_glosses[:10])
```

---

## Cell 5: Start Cloudflare Tunnel

```python
from pycloudflared import try_cloudflare
import threading
import time

print("🌐 Starting Cloudflare tunnel...")

tunnel_url = None

def start_tunnel():
    global tunnel_url
    result = try_cloudflare(port=8000, verbose=False)
    tunnel_url = result.tunnel
    
threading.Thread(target=start_tunnel, daemon=True).start()
time.sleep(10)

print("="*60)
print("🔗 TUNNEL URL (copy this!):")
print(f"   {tunnel_url}")
print("="*60)
print("\n⚠️ Update this URL in Cloudflare Pages environment variables!")
```

---

## Cell 6: Start Server (Runs Forever)

```python
print("🚀 Starting SignSpeak ASL Server...")
print("⚠️ This cell runs forever - that's normal!")
print("🛑 Press STOP button to shut down")
print("="*60)

!python -m uvicorn api.inference_server_wlasl:app --host 0.0.0.0 --port 8000 --reload
```

---

## Troubleshooting

### "Module not found"
```python
%cd /content/SignSpeak-Final-Year-Project/backend
```

### "Port already in use"
```python
!kill -9 $(lsof -t -i:8000) 2>/dev/null || true
```

### "Tunnel not working"
Re-run Cell 5 and update the URL in Cloudflare.

---

## Frontend Setup

1. **Cloudflare Dashboard** → Workers & Pages → Your project → Settings
2. **Environment Variables** → Add/Edit:
   - `COLAB_TUNNEL_URL` = Your tunnel URL from Cell 5 (no trailing slash!)
3. **Deployments** → Retry deployment

---

## Test

1. Visit: https://teamkathir.pages.dev/app
2. Allow camera
3. Make ASL signs
4. Watch for predictions!
