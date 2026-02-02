# Sign-to-Speech Deployment Guide

**Status:** Ready to deploy!  
**Last Updated:** February 2, 2026

---

## 📋 Pre-Deployment Checklist

### Required Files:
- ✅ `backend/best_isl_123.pth` (29.4 MB) - Your trained model
- ⚠️ `backend/checkpoints/label_mapping_123.json` - **YOU NEED TO CREATE THIS!**
- ✅ `backend/model.py` - Updated ✅
- ✅ `backend/api/inference_server.py` - Updated ✅
- ✅ Frontend code - Updated ✅

---

## STEP 1: Create Label Mapping File

### Where is your `file_to_label.json`?

Your training notebook created this file. It looks like:
```json
{
  "video_001": "hello",
  "video_002": "goodbye",
  "video_003": "hello",
  ...
}
```

### Convert it to `label_mapping_123.json`:

**Run this Python script:**

```python
import json
from collections import Counter

# Load your file_to_label.json from training
with open('file_to_label.json', 'r') as f:
    file_to_label = json.load(f)

# Get unique classes sorted alphabetically
unique_classes = sorted(set(file_to_label.values()))

# Create index mapping
label_mapping = {str(i): class_name for i, class_name in enumerate(unique_classes)}

# Save
with open('backend/checkpoints/label_mapping_123.json', 'w') as f:
    json.dump(label_mapping, f, indent=2)

print(f"✅ Created mapping with {len(label_mapping)} classes")
print("Classes:", list(label_mapping.values())[:10], "...")
```

**Expected output format:**
```json
{
  "0": "adult",
  "1": "alright",
  "2": "animal",
  ...
  "122": "yes"
}
```

---

## STEP 2: Deploy Backend

### Option A: Google Colab (FREE, Fast)

1. **Upload files to Colab:**
   - `backend/model.py`
   - `backend/api/inference_server.py`
   - `backend/checkpoints/best_isl_123.pth`
   - `backend/checkpoints/label_mapping_123.json`
   - `backend/requirements.txt`

2. **Install dependencies:**
```python
!pip install fastapi uvicorn torch pydantic python-multipart
```

3. **Run server:**
```python
!uvicorn inference_server:app --host 0.0.0.0 --port 8000 &
```

4. **Expose with Cloudflare Tunnel:**
```python
!npm install -g cloudflared
!cloudflared tunnel --url http://localhost:8000
```

5. **Copy the URL:**
```
https://abc-123-def.trycloudflare.com
```

### Option B: Render.com (FREE, Permanent)

1. Create `render.yaml`:
```yaml
services:
  - type: web
    name: signspeak-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn inference_server:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.12.0
```

2. Push `backend/` to GitHub
3. Connect GitHub repo to Render
4. Deploy automatically
5. Get URL: `https://signspeak-api.onrender.com`

### Option C: Railway (FREE, Simple)

1. Push `backend/` to GitHub
2. Go to railway.app
3. "New Project" → "Deploy from GitHub"
4. Select repo
5. Set start command: `uvicorn inference_server:app --host 0.0.0.0 --port $PORT`
6. Deploy!

---

## STEP 3: Deploy Frontend

### Update Environment Variable

**Create `.env.local`:**
```bash
# Replace with YOUR backend URL from Step 2
NEXT_PUBLIC_API_URL=https://your-backend-url.trycloudflare.com
```

### Test Locally First

```bash
npm install
npm run dev
```

Open `http://localhost:3000`:
- ✅ Click "Indian Sign Language"
- ✅ Allow camera
- ✅ Make hand gesture
- ✅ Should see prediction
- ✅ Should hear speech

### Push to GitHub (Auto-deploys to Cloudflare)

```bash
git add .
git commit -m "Complete sign-to-speech integration"
git push origin main
```

**Cloudflare Pages will:**
- Detect the push
- Build automatically
- Deploy to your domain
- ✅ Live in ~2 minutes!

### Configure Cloudflare Environment

Go to Cloudflare Pages dashboard:
1. Your project → Settings → Environment Variables
2. Add: `NEXT_PUBLIC_API_URL` = `https://your-backend-url.com`
3. Redeploy

---

## STEP 4: Test Integration

### Test Checklist:

1. **Homepage:**
   - [ ] ISL shows "Active ✅"
   - [ ] ASL shows "Coming Soon"
   - [ ] Click ISL card → Opens app

2. **Camera:**
   - [ ] Camera activates
   - [ ] MediaPipe extracts landmarks
   - [ ] Skeleton overlay visible (if enabled)

3. **Predictions:**
   - [ ] Make "hello" gesture
   - [ ] Prediction appears on screen
   - [ ] Confidence shown (e.g., "hello (85%)")
   - [ ] Prediction added to transcript

4. **Text-to-Speech:**
   - [ ] Prediction is spoken aloud
   - [ ] Voice is clear
   - [ ] Can hear each prediction

5. **Continuous:**
   - [ ] Make multiple gestures in sequence
   - [ ] Each is predicted and spoken
   - [ ] Transcript builds up

---

## STEP 5: Troubleshooting

### Problem: "Network error"
**Solution:** Check API URL in `.env.local` and Cloudflare env vars

### Problem: "CORS error"
**Solution:** Update backend `inference_server.py`:
```python
allow_origins=["*"]  # Or specific domain
```

### Problem: "Model not loaded"
**Solution:** 
- Check `best_isl_123.pth` is in `checkpoints/`
- Check `label_mapping_123.json` exists
- View backend logs for errors

### Problem: "No speech"
**Solution:**
- Ensure HTTPS (Web Speech API requires it)
- Check browser console for errors
- Try different browser (Chrome works best)

### Problem: "Wrong predictions"
**Solution:**
- Verify label_mapping_123.json is correct
- Check model loads without errors
- Test with known signs

---

## 🎯 Success Criteria

When working correctly:

1. ✅ Opens camera smoothly
2. ✅ Detects hands with MediaPipe
3. ✅ Sends landmarks to backend
4. ✅ Backend responds in <500ms
5. ✅ Prediction appears on UI
6. ✅ Prediction is spoken aloud
7. ✅ Continuous signing works
8. ✅ 123 ISL signs recognized

---

## 📊 Performance Targets

| Metric | Target | Acceptable |
|--------|--------|------------|
| API Response Time | <200ms | <500ms |
| End-to-end Latency | <800ms | <1.5s |
| Prediction Accuracy | 76% | 70%+ |
| Camera FPS | 30 | 20+ |

---

## Next Steps After Deployment

1. **Create demo video** showing the system in action
2. **Test with all 123 signs** systematically
3. **Collect user feedback** from testing
4. **Monitor performance** (latency, accuracy)
5. **Document known issues** for presentation

---

**Need Help?**
- Backend issues: Check Colab/Render logs
- Frontend issues: Check browser console
- Integration issues: Check network tab

**Ready to deploy!** 🚀
