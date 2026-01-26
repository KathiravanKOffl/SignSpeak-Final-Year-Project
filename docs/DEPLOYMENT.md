# SignSpeak Deployment Guide

Complete guide to deploy SignSpeak to Cloudflare Pages with zero cost.

## 📋 Prerequisites

- GitHub account
- Cloudflare account (free tier)
- Google Colab account (for ML backend)
- Node.js 18+ installed locally (for testing)

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

### Step 2: Connect to Cloudflare Pages

1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. Click **Workers & Pages** → **Create application** → **Pages** → **Connect to Git**
3. Authorize GitHub and select your repository
4. Click **Begin setup**

### Step 3: Configure Build Settings

**CRITICAL**: Use these exact settings:

- **Project name**: `your-project-name` (e.g., `signspeak`, `teamkathir`)
- **Production branch**: `main`
- **Build command**: `npm install && npm run pages:build`
- **Build output directory**: `.vercel/output/static`
- **Root directory**: `/`

> **Note**: The `pages:build` script runs `@cloudflare/next-on-pages` adapter which converts Next.js to Cloudflare Pages format.

### Step 4: Add Compatibility Flag

1. After first deployment, go to **Settings** → **Functions**
2. Scroll to **Compatibility flags**
3. Under **Production compatibility flags**:
   - Click **Configure Production compatibility flag**
   - Type: `nodejs_compat`
   - Click **Save**
4. Repeat for **Preview compatibility flags** (recommended)

### Step 5: Retry Deployment

1. Go to **Deployments** tab
2. Click on the latest deployment
3. Click **Retry deployment**
4. Wait for build to complete (~2-3 minutes)

### Step 6: Verify Deployment

Visit your site: `https://your-project-name.pages.dev`

You should see the SignSpeak landing page. ✅

---

## 🧪 Part 2: Backend Deployment (Google Colab)

### Step 1: Open Colab Notebook

1. Open `backend/colab_deployment.ipynb` in Google Colab
2. Or create new notebook and copy the code

### Step 2: Enable GPU

1. Click **Runtime** → **Change runtime type**
2. Select **T4 GPU**
3. Click **Save**

### Step 3: Run All Cells

Execute cells in order:
1. Install dependencies
2. Load model weights
3. Start FastAPI server
4. Create Cloudflare Tunnel

### Step 4: Copy Tunnel URL

After tunnel starts, you'll see output like:
```
Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):
https://abc-xyz-123.trycloudflare.com
```

Copy this URL. ✅

---

## ⚙️ Part 3: Connect Frontend to Backend

### Step 1: Add Environment Variables

1. Go to Cloudflare Dashboard → Your Pages project
2. **Settings** → **Environment variables**
3. Add the following:

**Production variables**:
```
COLAB_TUNNEL_URL = https://your-tunnel-url.trycloudflare.com
CLOUDFLARE_ACCOUNT_ID = your-cloudflare-account-id
CLOUDFLARE_API_TOKEN = your-api-token
```

**Preview variables**: (Same values)

### Step 2: Get Cloudflare Credentials

**Account ID**:
- Found in Cloudflare Dashboard → Workers & Pages → (Your project) → Settings, under "Account ID"

**API Token**:
1. Go to Cloudflare Dashboard → **My Profile** → **API Tokens**
2. Click **Create Token**
3. Use template: **Edit Cloudflare Workers**
4. Or create custom with: `Account.Workers AI:Read`
5. Copy the token immediately (shown only once)

### Step 3: Redeploy

1. Go to **Deployments** tab
2. Click **Retry deployment** on latest
3. Or push a new commit to trigger auto-deploy

---

## 🎯 Part 4: Verification & Testing

### Test Checklist

- [ ] Landing page loads at `https://your-project.pages.dev`
- [ ] Click ISL or ASL button
- [ ] Camera permission dialog appears
- [ ] Camera turns on and stays on
- [ ] MediaPipe models load (see console: "Graph successfully started running")
- [ ] Camera feed appears with "Live" indicator
- [ ] No console errors

### Common Issues

**404 Error**:
- Check build output directory is `.vercel/output/static`
- Verify build command is `npm run pages:build`

**Node.js Compatibility Error**:
- Add `nodejs_compat` flag in Settings → Functions → Compatibility flags

**Camera turns off after 3 seconds**:
- This is a known bug, fix pending
- See `camera_analysis.md` for details

**MediaPipe models don't load**:
- Check browser console for errors
- Verify WASM files are being served correctly
- Try hard refresh (Ctrl+F5)

---

## 📊 Cost Breakdown

| Service | Free Tier | Estimated Usage | Cost |
|---------|-----------|-----------------|------|
| Cloudflare Pages | 500 builds/month | ~10-20/month | **$0** |
| Cloudflare Workers AI | 10,000 requests/day | ~100/day | **$0** |
| Google Colab | T4 GPU 12 hrs/session | As needed | **$0** |
| GitHub | Unlimited repos | 1 repo | **$0** |
| **TOTAL** | | | **$0/month** |

---

## 🔄 Updating the Application

### Frontend Updates

```bash
# Make your changes
git add .
git commit -m "Your changes"
git push origin main
```

Cloudflare Pages auto-deploys on every push to `main`. ✅

### Backend Updates

1. Update `backend/` code locally
2. Commit and push to GitHub
3. Re-run Colab notebook cells
4. Update `COLAB_TUNNEL_URL` if tunnel URL changed

---

## 🛠️ Troubleshooting

### Build Fails

1. **Check build logs** in Cloudflare Dashboard → Deployments → (Failed build) → View logs
2. Common fixes:
   - Ensure all dependencies are in `dependencies`, not `devDependencies`
   - Verify `@cloudflare/next-on-pages` is installed
   - Check Next.js version compatibility

### Runtime Errors

1. **Open browser console** (F12)
2. Check for:
   - CORS errors → Backend not responding
   - 404s → Missing static files
   - TypeError → JS bundle issue

### Colab Session Expires

Google Colab free tier sessions expire after 12 hours of inactivity.

**Solution**:
1. Re-run all cells in notebook
2. Copy new tunnel URL
3. Update `COLAB_TUNNEL_URL` in Cloudflare
4. Retry deployment

---

## 📚 Additional Resources

- [Cloudflare Pages Docs](https://developers.cloudflare.com/pages/)
- [@cloudflare/next-on-pages](https://github.com/cloudflare/next-on-pages)
- [Next.js Edge Runtime](https://nextjs.org/docs/app/building-your-application/rendering/edge-and-nodejs-runtimes)
- [MediaPipe Tasks Vision](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)

---

## 🎓 Project Structure

```
SignSpeak/
├── app/                    # Next.js App Router
│   ├── api/               # Edge API routes
│   └── (pages)/           # UI pages
├── components/            # React components
├── hooks/                 # Custom hooks (MediaPipe)
├── backend/               # Python ML backend
│   └── colab_deployment.ipynb
└── docs/                  # Documentation (you are here!)
```

---

## ✅ Success Criteria

Your deployment is successful when:

1. ✅ Site loads at Cloudflare Pages URL
2. ✅ No build or runtime errors
3. ✅ Camera initializes and captures video
4. ✅ MediaPipe models load in browser
5. ✅ Backend API is accessible (when Colab running)

---

**Questions or issues?** Check `CLOUDFLARE_ARCHITECTURE.md` for system architecture details.
