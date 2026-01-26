# SignSpeak - Complete Deployment Guide

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Part A: Cloudflare Setup (First Time)](#part-a-cloudflare-setup-first-time)
3. [Part B: Deploy Frontend to Cloudflare Pages](#part-b-deploy-frontend-to-cloudflare-pages)
4. [Part C: Deploy Backend to Google Colab](#part-c-deploy-backend-to-google-colab)
5. [Part D: Connect Frontend to Backend](#part-d-connect-frontend-to-backend)
6. [Testing & Verification](#testing--verification)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### What You Need:
- ✅ **Email address** (for Cloudflare and Google accounts)
- ✅ **GitHub account** (free)
- ✅ **Google account** (for Colab - free)
- ✅ **Web browser** (Chrome, Firefox, or Edge)
- ❌ **NO credit card required**
- ❌ **NO installation needed** (everything is cloud-based)

### Knowledge Level:
- This guide assumes **zero prior Cloudflare experience**
- We'll walk through every single click
- Screenshots and explanations provided

---

## Part A: Cloudflare Setup (First Time)

### Step A1: Create Cloudflare Account

1. **Go to Cloudflare**
   - Open browser → Visit: https://dash.cloudflare.com/sign-up
   
2. **Sign Up**
   - Enter your email address
   - Create a strong password
   - Click "Create Account"
   - Check your email for verification link
   - Click the link to verify your account

3. **Skip Domain Setup** (if prompted)
   - Cloudflare may ask "Add a site"
   - Click "Skip" or "I'll do this later"
   - We don't need a custom domain - Pages gives us a free one!

4. **You're In!**
   - You should now see the Cloudflare dashboard
   - URL should be: `https://dash.cloudflare.com/`

---

### Step A2: Get Your Account ID

Your Account ID is a unique identifier needed for API calls.

1. **Navigate to Workers & Pages**
   - In the left sidebar, click "Workers & Pages"
   - Or go directly to: https://dash.cloudflare.com/?to=/:account/workers-and-pages

2. **Find Your Account ID**
   - On the right side of the page, you'll see a section called "Account details"
   - Look for "Account ID"
   - It's a long string like: `a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6`
   
3. **Copy Your Account ID**
   - Click the "Click to copy" button next to your Account ID
   - **SAVE THIS!** Open a text file and paste it:
     ```
     CLOUDFLARE_ACCOUNT_ID=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
     ```

💡 **Alternative Method**:
- Look at your browser URL
- After `/`, you'll see your account ID
- Example URL: `https://dash.cloudflare.com/a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6/workers-and-pages`
- The Account ID is the long string after the first `/`

---

### Step A3: Create API Token

API tokens allow your app to use Cloudflare Workers AI.

1. **Go to API Tokens Page**
   - Click your profile icon (top right)
   - Select "My Profile"
   - Click "API Tokens" in the left sidebar
   - Or go directly to: https://dash.cloudflare.com/profile/api-tokens

2. **Create Custom Token**
   - Click "Create Token" button
   - Look for "Create Custom Token" section
   - Click "Get started" button

3. **Configure Token Permissions**
   
   **Token Name**:
   ```
   SignSpeak Workers AI Token
   ```
   
   **Permissions** (Add these):
   
   | Permission | Resource | Access Level |
   |------------|----------|--------------|
   | Account | Workers AI | Read |
   | Account | Account Settings | Read |
   
   **To add permissions**:
   - Click "+ Add more" under Permissions
   - For each row:
     - First dropdown: Select "Account"
     - Second dropdown: Select "Workers AI" (first one), then "Account Settings" (second one)
     - Third dropdown: Select "Read"

4. **Set Account Resources**
   - Under "Account Resources"
   - Select "Include" → "Specific account"
   - Choose your account from dropdown (should be the only one)

5. **Client IP Address Filtering** (Optional)
   - Leave blank for "All IP addresses"
   - Or restrict to your IP for extra security

6. **TTL (Time to Live)**
   - Leave as default or set to longer duration
   - Recommended: No expiration (for development)

7. **Create Token**
   - Click "Continue to summary"
   - Review your settings
   - Click "Create Token"

8. **SAVE YOUR TOKEN!** ⚠️
   - You'll see a screen with your token
   - It looks like: `cloudflare_token_abc123xyz789...`
   - **Copy it immediately** - you can't see it again!
   - **SAVE THIS!** Add to your text file:
     ```
     CLOUDFLARE_API_TOKEN=cloudflare_token_abc123xyz789...
     ```
   
9. **Test Your Token** (Optional but Recommended)
   - Cloudflare provides a test command
   - Copy and run it in your terminal:
     ```bash
     curl -X GET "https://api.cloudflare.com/client/v4/user/tokens/verify" \
          -H "Authorization: Bearer YOUR_TOKEN_HERE"
     ```
   - Should return: `"status": "active"`

---

### Step A4: Install Wrangler CLI (Optional for Manual Deployment)

If you want to deploy manually from your computer:

1. **Install Node.js** (if not already installed)
   - Go to: https://nodejs.org/
   - Download LTS version
   - Install with default settings

2. **Install Wrangler**
   ```bash
   npm install -g wrangler
   ```

3. **Login to Cloudflare**
   ```bash
   wrangler login
   ```
   - Browser window will open
   - Click "Allow" to authorize
   - Terminal will show "Successfully logged in"

4. **Verify Installation**
   ```bash
   wrangler whoami
   ```
   - Should show your Cloudflare account email

---

## Part B: Deploy Frontend to Cloudflare Pages

We'll use **GitHub Integration** (recommended - easiest!).

### Step B1: Prepare GitHub Repository

Your code is already on GitHub! ✅
- Repository: https://github.com/KathiravanKOffl/SignSpeak-Final-Year-Project
- Branch: `main`

---

### Step B2: Connect Cloudflare Pages to GitHub

1. **Go to Cloudflare Pages**
   - From Cloudflare dashboard: https://dash.cloudflare.com/
   - Left sidebar → Click "Workers & Pages"
   - Click "Create application" button
   - Select "Pages" tab
   - Click "Connect to Git"

2. **Connect GitHub**
   - Click "Connect GitHub" button
   - A GitHub authorization page will open
   - Click "Authorize Cloudflare-Pages"
   - You may need to enter your GitHub password

3. **Select Repository**
   - You'll see a list of your repositories
   - Find and select: `SignSpeak-Final-Year-Project`
   - Click "Begin setup"

---

### Step B3: Configure Build Settings

1. **Project Name**
   ```
   signspeak
   ```
   - This will be your subdomain: `signspeak.pages.dev`

2. **Production Branch**
   ```
   main
   ```

3. **Framework Preset**
   - Select: "Next.js (Advanced)"
   - Or select "None" if not listed

4. **Build Command**
   ```
   cd signspeak-web && npm install && npm run pages:build
   ```

5. **Build Output Directory**
   ```
   signspeak-web/.vercel/output/static
   ```

6. **Root Directory**
   - Click "Advanced" if not visible
   - Set root directory to:
   ```
   /
   ```
   (Leave as root, we handle path in build command)

7. **Environment Variables** (IMPORTANT!)
   
   Click "Add variable" for each of these:
   
   | Variable Name | Value | Notes |
   |---------------|-------|-------|
   | `NODE_ENV` | `production` | Required |
   | `CLOUDFLARE_ACCOUNT_ID` | `your_account_id_from_step_a2` | Paste from Step A2 |
   | `CLOUDFLARE_API_TOKEN` | `your_api_token_from_step_a3` | Paste from Step A3 |
   | `COLAB_TUNNEL_URL` | `https://temp.trycloudflare.com` | We'll update this later in Part D |

   **To add each variable**:
   - Click "+ Add variable"
   - Enter variable name in left box
   - Enter value in right box
   - Repeat for all 4 variables

---

### Step B4: Deploy!

1. **Click "Save and Deploy"**
   - Cloudflare will start building your app
   - You'll see a build log with progress

2. **Wait for Build** (2-5 minutes)
   - Green checkmarks appear for each step:
     - ✅ Cloning repository
     - ✅ Installing dependencies
     - ✅ Building project
     - ✅ Deploying to Cloudflare's network

3. **Build Success!** 🎉
   - You'll see: "Success! Deployed to [URL]"
   - Your app is now live at: `https://signspeak.pages.dev`
   - Click "Continue to project"

---

### Step B5: Verify Frontend Deployment

1. **Visit Your App**
   - Open: `https://signspeak.pages.dev`
   - You should see the SignSpeak landing page
   - Language selection (ISL/ASL) should be visible

2. **Check Deployment Details**
   - In Cloudflare dashboard → Pages → signspeak
   - You'll see:
     - Deployment status: Active ✅
     - Production branch: main
     - Last deployed: Just now
     - Visit site button

3. **Test Basic Functionality**
   - Click on ISL or ASL language
   - You should reach the app page
   - Camera permission popup may appear (OK to allow or deny for now)
   - Backend won't work yet (we haven't deployed Colab)

---

## Part C: Deploy Backend to Google Colab

### Step C1: Open Google Colab

1. **Go to Google Colab**
   - Visit: https://colab.research.google.com/
   - Sign in with your Google account (if not already)

2. **Create New Notebook**
   - Click "New notebook" or
   - File → New notebook

3. **Name Your Notebook**
   - Click "Untitled" at the top
   - Rename to: `SignSpeak Inference Server`

---

### Step C2: Enable GPU

1. **Change Runtime Type**
   - Top menu → Runtime → Change runtime type
   - In the popup:
     - Runtime type: Python 3
     - Hardware accelerator: **T4 GPU** ⚡
   - Click "Save"

2. **Verify GPU**
   - Add this code to first cell:
   ```python
   !nvidia-smi
   ```
   - Click ▶ play button or press Shift+Enter
   - Should show "Tesla T4" GPU info

---

### Step C3: Setup Backend Code

Copy and paste each cell one by one:

**Cell 1: Install Dependencies**
```python
print("📦 Installing dependencies...")

!pip install -q torch torchvision torchaudio
!pip install -q fastapi uvicorn pydantic python-multipart
!pip install -q pycloudflared

print("✅ Dependencies installed!")
```

**Cell 2: Clone Repository**
```python
import os

print("📥 Cloning repository...")

# Remove if exists
if os.path.exists('SignSpeak-Final-Year-Project'):
    !rm -rf SignSpeak-Final-Year-Project

!git clone https://github.com/KathiravanKOffl/SignSpeak-Final-Year-Project.git

%cd SignSpeak-Final-Year-Project/backend

print("✅ Repository cloned!")
print(f"Current directory: {os.getcwd()}")
```

**Cell 3: Verify GPU**
```python
import torch

print(f"🖥️  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"💾 CUDA Available: {torch.cuda.is_available()}")
print(f"🔥 PyTorch Version: {torch.__version__}")
```

**Cell 4: Start Cloudflare Tunnel** (MOST IMPORTANT!)
```python
from pycloudflared import try_cloudflare
import threading
import time

print("🌐 Starting Cloudflare Tunnel...")

tunnel_url = None

def start_tunnel():
    global tunnel_url
    try:
        url_obj = try_cloudflare(port=8000, verbose=True)
        tunnel_url = url_obj.tunnel
        print(f"\n" + "="*60)
        print(f"✅ Tunnel Active!")
        print(f"📋 COPY THIS URL:")
        print(f"    {tunnel_url}")
        print(f"="*60)
        print(f"\n🔧 Next Step: Update Cloudflare Pages environment variable!")
        print(f"   Variable: COLAB_TUNNEL_URL")
        print(f"   Value: {tunnel_url}")
        print("="*60)
    except Exception as e:
        print(f"❌ Tunnel error: {e}")

# Start tunnel in background thread
tunnel_thread = threading.Thread(target=start_tunnel, daemon=True)
tunnel_thread.start()

# Wait for tunnel to start
print("⏳ Waiting for tunnel...")
time.sleep(10)

if tunnel_url:
    print(f"✅ Ready! Tunnel URL: {tunnel_url}")
else:
    print("⏳ Tunnel starting... Check output above for URL")
```

**IMPORTANT**: After running Cell 4, you'll see output like:
```
✅ Tunnel Active!
📋 COPY THIS URL:
    https://abc-xyz-123-def.trycloudflare.com
```

**📋 COPY THAT URL!** Write it down or keep the tab open.

**Cell 5: Start FastAPI Server**
```python
print("🚀 Starting FastAPI inference server...")
print("⚠️  This cell will run indefinitely - that's normal!")
print("📝 Check logs below for server status")
print("="*60)

!python -m uvicorn api.inference_server:app --host 0.0.0.0 --port 8000 --reload
```

This cell will run forever (that's correct!). You'll see logs like:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Started server process
INFO:     Waiting for application startup
INFO:     Application startup complete
```

**Cell 6: Keep-Alive** (Optional - Run in another cell if Colab disconnects)
```python
"""
Keep-Alive Script
Run this to prevent Colab disconnection
"""
import time
from IPython.display import clear_output

print("⏰ Keep-alive active...")
counter = 0

try:
    while True:
        counter += 1
        clear_output(wait=True)
        print(f"⏰ Server running: {counter} minutes")
        print(f"🌐 Tunnel URL: {tunnel_url if tunnel_url else 'Check Cell 4'}")
        print(f"💚 Colab session: ACTIVE")
        print(f"\n💡 Tip: Keep this tab open to prevent disconnection")
        time.sleep(60)
except KeyboardInterrupt:
    print("Stopped.")
```

---

### Step C4: Test Backend

1. **Check Server Logs** (in Cell 5 output)
   - Should show: "Application startup complete"
   - No error messages

2. **Test Tunnel URL** (in new browser tab)
   - Open the tunnel URL from Cell 4
   - Example: `https://abc-xyz-123.trycloudflare.com`
   - You should see: `{"status": "online", "model_loaded": true, ...}`

3. **Test Health Endpoint**
   - Add `/health` to your tunnel URL
   - Example: `https://abc-xyz-123.trycloudflare.com/health`
   - Should return JSON with model status

---

## Part D: Connect Frontend to Backend

### Step D1: Update Cloudflare Pages Environment Variable

1. **Go to Cloudflare Pages Settings**
   - Cloudflare dashboard: https://dash.cloudflare.com/
   - Workers & Pages → signspeak
   - Click "Settings" tab
   - Scroll to "Environment variables"

2. **Edit COLAB_TUNNEL_URL**
   - Find the variable `COLAB_TUNNEL_URL`
   - Click "Edit" (pencil icon)
   - **Replace** `https://temp.trycloudflare.com`
   - **With** your actual tunnel URL from Step C3
   - Example: `https://abc-xyz-123-def.trycloudflare.com`
   - Click "Save"

3. **Important**: 
   - Make sure there's NO trailing slash
   - ✅ Correct: `https://abc-xyz.trycloudflare.com`
   - ❌ Wrong: `https://abc-xyz.trycloudflare.com/`

---

### Step D2: Redeploy Frontend

After updating environment variables, you need to redeploy:

1. **Trigger New Deployment**
   - Cloudflare dashboard → signspeak project
   - Click "Deployments" tab
   - Click "View build" on the latest deployment
   - Click "Redeploy" button
   - Confirm: "Redeploy to production"

2. **Wait for Build** (2-3 minutes)

3. **Verify New Environment Variable**
   - After deployment completes
   - Settings → Environment variables
   - `COLAB_TUNNEL_URL` should show your tunnel URL

## 🚀 Cloudflare Pages Deployment

SignSpeak is designed to deploy entirely on Cloudflare's free tier infrastructure.

### Prerequisites

1. **Cloudflare Account** (free tier is sufficient)
   - Sign up at [cloudflare.com](https://cloudflare.com)
   - Get your Account ID from the dashboard
   - Create an API token with Workers AI permissions

2. **Wrangler CLI**
```bash
npm install -g wrangler
wrangler login
```

3. **Google Colab** (for ML backend)
   - Free T4 GPU access
   - No credit card required

---

## 📦 Deployment Steps

### Step 1: Configure Environment Variables

1. Go to Cloudflare Pages dashboard
2. Select your project (or create new)
3. Go to Settings → Environment Variables
4. Add the following:

```env
# Required
CLOUDFLARE_ACCOUNT_ID=your_account_id_here
CLOUDFLARE_API_TOKEN=your_api_token_here

# Will be set after Colab deployment
COLAB_TUNNEL_URL=https://your-tunnel.trycloudflare.com

# Optional
NODE_ENV=production
NEXT_PUBLIC_APP_URL=https://your-app.pages.dev
```

### Step 2: Deploy Frontend to Cloudflare Pages

#### Option A: GitHub Integration (Recommended)

1. Push code to GitHub (already done!)
2. Go to Cloudflare Pages dashboard
3. Click "Create a project"
4. Connect to GitHub: `KathiravanKOffl/SignSpeak-Final-Year-Project`
5. Configure build:
   - **Build command**: `npm run pages:build`
   - **Build output directory**: `.vercel/output/static`
   - **Root directory**: `signspeak-web`
6. Click "Save and Deploy"
7. Add environment variables in dashboard

#### Option B: Manual Deployment

```bash
cd signspeak-web

# Install dependencies
npm install

# Build for Cloudflare Pages
npm run pages:build

# Deploy
npx wrangler pages deploy

# Or use the combined script
npm run pages:deploy
```

### Step 3: Deploy Backend to Google Colab

1. Open the Colab notebook:
   - `backend/colab_deployment.ipynb`
   - Or create new notebook and paste cells

2. Run cells in order:
   - Cell 1: Install dependencies
   - Cell 2: Clone repository
   - Cell 3: Check GPU
   - Cell 4: Start Cloudflare Tunnel
   - Cell 5: Start FastAPI server
   - Cell 6: Keep-alive script

3. Copy the tunnel URL (e.g., `https://abc-xyz-123.trycloudflare.com`)

4. Update Cloudflare Pages environment variable:
   - Go to Pages dashboard
   - Settings → Environment Variables
   - Update `COLAB_TUNNEL_URL` with the tunnel URL
   - Trigger new deployment (or wait for auto-deploy)

### Step 4: Verify Deployment

1. Visit your Cloudflare Pages URL
2. Check `/api/predict` health endpoint
3. Test camera permissions
4. Verify backend connection

---

## 🔧 Configuration Files

### `wrangler.toml`
Cloudflare Pages configuration file defining:
- Environment variables
- Build settings
- Compatibility date
- Future: Durable Objects, KV namespaces

### `package.json` Scripts
- `npm run dev` - Local development
- `npm run build` - Standard Next.js build
- `npm run pages:build` - Build for Cloudflare Pages
- `npm run pages:deploy` - Build + deploy to Pages
- `npm run pages:dev` - Local Cloudflare Pages emulation

---

## 🌐 URLs After Deployment

### Frontend (Cloudflare Pages)
```
Production: https://signspeak.pages.dev
Preview: https://[branch].signspeak.pages.dev
```

### Backend (Google Colab + Tunnel)
```
Tunnel: https://[random].trycloudflare.com
Endpoints:
  - GET  /health
  - POST /predict
  - GET  /
```

### API Routes (Proxied through Pages)
```
- POST /api/predict     → Colab backend
- POST /api/transcribe  → Cloudflare Workers AI (Whisper)
- POST /api/translate   → Cloudflare Workers AI (Llama-3)
- POST /api/room        → Room creation
- GET  /api/room?room=X → Room info
```

---

## 💰 Cost Breakdown (Zero!)

| Service | Free Tier Limit | Usage | Cost |
|---------|----------------|-------|------|
| Cloudflare Pages | 500 builds/month | ~10 builds/month | $0 |
| Cloudflare Workers AI | 10,000 requests/day | ~100 requests/day | $0 |
| Google Colab | 12 hours/session | Running as needed | $0 |
| Cloudflare Tunnel | Unlimited | 1 tunnel | $0 |
| **TOTAL** | | | **$0/month** |

---

## 🔐 Security Notes

1. **API Tokens**: Never commit tokens to git
2. **CORS**: Already configured for *.pages.dev
3. **Rate Limiting**: Consider adding for production
4. **Environment Variables**: Set in Cloudflare dashboard only

---

## 🐛 Troubleshooting

### "Backend server URL not configured"
- Check `COLAB_TUNNEL_URL` environment variable in Cloudflare dashboard
- Ensure Colab notebook is running
- Verify tunnel URL is correct

### "Cannot reach backend"
- Colab session may have timed out (restart notebook)
- Tunnel may have disconnected (check Cell 4 output)
- Check firewall/network restrictions

### "MediaPipe models not loading"
- This is normal in local development
- Models load from CDN - may take 5-10 seconds
- After Cloudflare deployment, CDN access is faster

### Build fails
```bash
# Clear cache and rebuild
rm -rf .next .vercel
npm install
npm run pages:build
```

---

## 📈 Monitoring

### Cloudflare Dashboard
- Real-time analytics
- Request rates
- Error logs
- Build history

### Colab Logs
- Cell output shows server logs
- Keep-alive cell shows uptime
- GPU usage visible in Colab UI

---

## 🔄 Continuous Deployment

### Enabled by Default (GitHub Integration)
1. Push to GitHub → Auto-deploy to Cloudflare Pages
2. Pull requests → Preview deployments
3. Main branch → Production deployment

### Manual Pipeline
```bash
# Local changes
git add -A
git commit -m "Your changes"
git push origin main

# Wait for Cloudflare Pages to auto-deploy
# Or manually trigger via dashboard
```

---

## 📝 Post-Deployment Checklist

- [ ] Frontend deployed to Cloudflare Pages
- [ ] Environment variables configured
- [ ] Backend running on Google Colab
- [ ] Tunnel URL updated in frontend
- [ ] Camera permissions working
- [ ] MediaPipe models loading
- [ ] API endpoints responding
- [ ] Multi-device mode functional
- [ ] Domain configured (optional)
- [ ] Analytics enabled (optional)

---

## 🎉 You're Live!

Your SignSpeak app is now:
- ✅ Deployed globally on Cloudflare's CDN
- ✅ Using zero-cost infrastructure
- ✅ Auto-scaling to handle traffic
- ✅ Privacy-preserving (edge processing)
- ✅ Supporting both ISL and ASL

**Share your app**: `https://signspeak.pages.dev`
