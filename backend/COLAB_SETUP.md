# Google Colab Backend Setup - Copy & Paste Guide

Simple guide to deploy SignSpeak backend on Google Colab. Just copy each code block into a new cell and run.

---

## 🚀 Quick Start

1. Open https://colab.research.google.com/
2. Create new notebook
3. **Runtime** → **Change runtime type** → **T4 GPU** → Save
4. Copy-paste each cell below in order
5. Run each cell (Shift+Enter)

---

## 📋 Cells to Create

### Cell 1: Install Dependencies

```python
print("📦 Installing dependencies...")

!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q fastapi uvicorn[standard] pydantic python-multipart
!pip install -q pycloudflared

print("✅ Dependencies installed!")
```

---

### Cell 2: Clone Repository

```python
import os

print("📥 Cloning repository...")

# Remove if exists
if os.path.exists('SignSpeak-Final-Year-Project'):
    !rm -rf SignSpeak-Final-Year-Project

!git clone https://github.com/KathiravanKOffl/SignSpeak-Final-Year-Project.git
%cd SignSpeak-Final-Year-Project/backend

print(f"✅ Repository cloned!")
print(f"📂 Current directory: {os.getcwd()}")
```

---

### Cell 3: Verify GPU

```python
import torch

print("🔍 Checking GPU...\n")
print(f"🖥️  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"💾 CUDA: {torch.cuda.is_available()}")
print(f"🔥 PyTorch: {torch.__version__}")

if torch.cuda.is_available():
    print(f"\n✅ GPU ready!")
else:
    print(f"\n⚠️  Using CPU (slower)")
```

---

### Cell 4: Start Cloudflare Tunnel ⚡ IMPORTANT

```python
from pycloudflared import try_cloudflare
import threading
import time

print("🌐 Starting Cloudflare Tunnel...\n")

tunnel_url = None

def start_tunnel():
    global tunnel_url
    try:
        url_obj = try_cloudflare(port=8000, verbose=True)
        tunnel_url = url_obj.tunnel
        print(f"\n" + "="*70)
        print(f"✅ TUNNEL ACTIVE!")
        print(f"\n📋 COPY THIS URL:")
        print(f"    {tunnel_url}")
        print(f"\n" + "="*70)
        print(f"\n🔧 Next Steps:")
        print(f"   1. Copy the URL above")
        print(f"   2. Go to Cloudflare Pages dashboard")
        print(f"   3. Settings → Environment variables")
        print(f"   4. Edit COLAB_TUNNEL_URL")
        print(f"   5. Paste this URL and save")
        print(f"   6. Redeploy your Pages project")
        print(f"\n" + "="*70)
    except Exception as e:
        print(f"❌ Error: {e}")

# Start in background
tunnel_thread = threading.Thread(target=start_tunnel, daemon=True)
tunnel_thread.start()

print("⏳ Waiting for tunnel...")
time.sleep(10)

if tunnel_url:
    print(f"\n✅ Ready! URL: {tunnel_url}")
else:
    print(f"\n⏳ Check output above for URL")
```

**⚠️ IMPORTANT**: Copy the tunnel URL that appears! You'll need it for Cloudflare.

---

### Cell 5: Start FastAPI Server

```python
print("🚀 Starting server...")
print("⚠️  This cell runs forever - that's normal!")
print("🛑 Press stop button to shut down")
print("="*70)

!python -m uvicorn api.inference_server:app --host 0.0.0.0 --port 8000 --reload
```

This cell will keep running. You'll see logs like:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Started server process
INFO:     Application startup complete
```

---

### Cell 6: Keep-Alive (Optional)

Run this in a **separate cell** if Colab tends to disconnect:

```python
import time
from IPython.display import clear_output

print("⏰ Keep-alive active...")
counter = 0

try:
    while True:
        counter += 1
        clear_output(wait=True)
        print(f"⏰ Uptime: {counter} minutes")
        print(f"🌐 Tunnel: {tunnel_url if tunnel_url else 'See Cell 4'}")
        print(f"💚 Status: ACTIVE")
        print(f"\n💡 Keep this tab open")
        time.sleep(60)
except KeyboardInterrupt:
    print("Stopped")
```

---

## ✅ Verification

After Cell 4 completes, open the tunnel URL in browser:
```
https://your-random-id.trycloudflare.com
```

You should see JSON response like:
```json
{
  "status": "online",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

## 🔗 Connect to Frontend

1. **Copy tunnel URL** from Cell 4 output
2. Go to **Cloudflare Dashboard**: https://dash.cloudflare.com/
3. **Workers & Pages** → **your-project** → **Settings**
4. **Environment variables** → Find `COLAB_TUNNEL_URL`
5. Click **Edit** → Paste tunnel URL → **Save**
6. **Deployments** tab → **Retry deployment**

---

## 🐛 Troubleshooting

### Cell 1 fails
- Check internet connection
- Wait 30 seconds and retry

### Cell 2 fails
- Repository might be private - check GitHub access
- Or repository moved - update URL

### Cell 4 shows no tunnel URL
- Wait 30 seconds, tunnel takes time
- Check Cell 4 output for errors
- Rerun Cell 4 if needed

### Cell 5 crashes
- Make sure you're in `backend/` directory (Cell 2)
- Check if `api/inference_server.py` exists
- Verify all dependencies installed (Cell 1)

### Colab disconnects
- Run Cell 6 (keep-alive)
- Keep browser tab active
- Upgrade to Colab Pro for longer sessions

---

## 📊 Resource Limits

**Free Tier**:
- 12 hours per session max
- T4 GPU (16GB VRAM)
- ~12GB RAM
- Will disconnect if idle

**When Session Expires**:
1. Runtime → Restart runtime
2. Run Cell 1-5 again
3. Copy new tunnel URL
4. Update Cloudflare environment variable
5. Redeploy

---

## 💡 Tips

- **Don't close the tab** - Colab disconnects
- **Run keep-alive** (Cell 6) to prevent idle timeout
- **Tunnel URL changes** each session - update Cloudflare
- **Test endpoint**: `your-tunnel-url/health`
- **Free tier**: 12 hours max, then restart

---

## 🎯 Next Steps After Setup

1. ✅ All 5 cells running
2. ✅ Tunnel URL copied
3. ✅ Cloudflare env var updated
4. ✅ Frontend redeployed
5. ✅ Test: Visit your Pages URL
6. ✅ Click language → Camera should work
7. ✅ Sign recognition should send to Colab backend

---

**Need Help?** Check full deployment guide in `docs/DEPLOYMENT.md`
