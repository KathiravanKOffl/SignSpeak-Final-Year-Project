# SignSpeak Deployment Guide

Complete guide to deploy SignSpeak to Cloudflare Pages + Google Colab backend with zero cost.

---

## 📋 Prerequisites

- GitHub account
- Cloudflare account (free tier)
- Google account (for Colab)
- Node.js 18+ (for local testing only)

---

## 🚀 Part 1: Frontend Deployment (Cloudflare Pages)

### Step 1: Push Code to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2: Create Cloudflare Pages Project

1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. **Workers & Pages** → **Create application** → **Pages** → **Connect to Git**
3. Authorize GitHub and select your repository
4. Click **Begin setup**

### Step 3: Configure Build Settings ⚠️ CRITICAL

| Setting | Value |
|---------|-------|
| **Project name** | `your-project-name` |
| **Production branch** | `main` |
| **Build command** | `npm install && npm run pages:build` |
| **Build output directory** | `.vercel/output/static` |
| **Root directory** | `/` |

> **Important**: The `pages:build` script runs `@cloudflare/next-on-pages` adapter which converts Next.js to Cloudflare Pages format.

### Step 4: Add Compatibility Flag ⚠️ REQUIRED

After first deployment (will fail without this):

1. Go to **Settings** → **Functions** (scroll down)
2. Find **Compatibility flags** section
3. **Production compatibility flags**: Add `nodejs_compat`
4. **Preview compatibility flags**: Add `nodejs_compat`
5. Click **Save**

### Step 5: Retry Deployment

1. Go to **Deployments** tab
2. Click latest (failed) deployment → **Retry deployment**
3. Wait ~2-3 minutes for build

### Step 6: Verify

Visit: `https://your-project-name.pages.dev`

---

## 🧪 Part 2: Backend Deployment (Google Colab)

### Step 1: Create Colab Notebook

1. Go to https://colab.research.google.com/
2. Create new notebook
3. **Runtime** → **Change runtime type** → **T4 GPU** → Save

### Step 2: Run These Cells

**Cell 1: Install Dependencies**
```python
print("📦 Installing dependencies...")
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q fastapi uvicorn[standard] pydantic python-multipart
!pip install -q pycloudflared
print("✅ Done!")
```

**Cell 2: Clone Repository**
```python
import os
if os.path.exists('SignSpeak-Final-Year-Project'):
    !rm -rf SignSpeak-Final-Year-Project
!git clone https://github.com/KathiravanKOffl/SignSpeak-Final-Year-Project.git
%cd SignSpeak-Final-Year-Project/backend
print(f"✅ Ready! Dir: {os.getcwd()}")
```

**Cell 3: Verify GPU**
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"CUDA: {torch.cuda.is_available()}")
```

**Cell 4: Start Tunnel** ⚠️ COPY THE URL!
```python
from pycloudflared import try_cloudflare
import threading, time

tunnel_url = None
def start_tunnel():
    global tunnel_url
    url_obj = try_cloudflare(port=8000, verbose=True)
    tunnel_url = url_obj.tunnel
    print(f"\n{'='*60}\n✅ TUNNEL URL:\n    {tunnel_url}\n{'='*60}")

threading.Thread(target=start_tunnel, daemon=True).start()
time.sleep(10)
print(f"URL: {tunnel_url}" if tunnel_url else "Starting...")
```

**Cell 5: Start Server**
```python
!python -m uvicorn api.inference_server:app --host 0.0.0.0 --port 8000 --reload
```

---

## ⚙️ Part 3: Connect Frontend to Backend

### Step 1: Add Environment Variables

In Cloudflare Dashboard → Your project → **Settings** → **Environment variables**:

| Variable | Value |
|----------|-------|
| `COLAB_TUNNEL_URL` | `https://your-tunnel.trycloudflare.com` |
| `CLOUDFLARE_ACCOUNT_ID` | Your account ID (from dashboard URL) |
| `CLOUDFLARE_API_TOKEN` | Create at Profile → API Tokens |

### Step 2: Redeploy

**Deployments** → Click latest → **Retry deployment**

---

## ✅ Part 4: Verification

### Test Checklist

- [ ] Landing page loads at your Pages URL
- [ ] Click ISL or ASL → Camera permission requested
- [ ] Camera light ON and video feed visible
- [ ] Console shows: `[Camera] ✅ Video ready with enough data!`
- [ ] Console shows: `Graph successfully started running.` (MediaPipe)
- [ ] Backend accessible at tunnel URL → Shows JSON status

### Console Logs (Expected)

```
[Camera] Requesting camera access...
[Camera] Camera access granted
[Camera] Stream attached to video element
[Camera] Video can play, starting...
[Camera] Video playing!
[Camera] ✅ Video ready with enough data!
[Camera] Starting frame processing...
```

---

## 🐛 Troubleshooting

### Build Fails with "edge runtime" error
- Ensure all API routes have: `export const runtime = 'edge';`
- Located in: `app/api/*/route.ts`

### "Node.JS Compatibility Error" on site
- Add `nodejs_compat` flag in Settings → Functions → Compatibility flags

### 404 Error
- Check output directory is `.vercel/output/static`
- Verify build command is `npm run pages:build`

### Camera stuck on "Starting video..."
- This was a bug (video element not in DOM during loading)
- Fixed in commit `21f70dd` - ensure you have latest code

### Backend 502 Bad Gateway
- Check Colab notebook is still running
- Verify Cell 5 (server) is active
- Check tunnel URL is correct in env vars

### Colab Session Expires
- Free tier: 12 hours max
- Re-run all cells, copy new tunnel URL
- Update Cloudflare env var, redeploy

---

## 💰 Cost Breakdown

| Service | Free Tier | Cost |
|---------|-----------|------|
| Cloudflare Pages | 500 builds/month | $0 |
| Cloudflare Workers AI | 10,000 req/day | $0 |
| Google Colab | T4 GPU, 12hr sessions | $0 |
| **TOTAL** | | **$0/month** |

---

## 🔄 Updating

### Frontend Changes
```bash
git add .
git commit -m "Your changes"
git push origin main
# Auto-deploys to Cloudflare Pages
```

### Backend Changes
```bash
git push origin main
# In Colab: %cd /content && !rm -rf SignSpeak* && <run Cell 2 again>
```

---

## 📚 Project URLs

- **Frontend**: `https://teamkathir.pages.dev`
- **Backend Health**: `https://[tunnel-url]/health`
- **API Predict**: `POST https://[tunnel-url]/predict`

---

## 🎓 Current Status

| Feature | Status |
|---------|--------|
| Camera capture | ✅ Operational |
| MediaPipe detection | ✅ Operational |
| Backend server | ✅ Active |
| Tunnel connection | ✅ Secure |
| ML Model | ✅ Trained & Optimized |
| Sign recognition | ✅ Production Ready |
