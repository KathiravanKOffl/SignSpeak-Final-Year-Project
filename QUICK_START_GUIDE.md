# 🔧 Fix Kaggle Training & Test App - Quick Guide

## Part 1: Restart Kaggle Training (5 minutes setup)

### Step 1: Create New Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **"+ New Notebook"**
3. Name it: `SignSpeak Training V2`

### Step 2: Add Dataset

1. Click **"+ Add data"** (right sidebar)
2. Search: **"land-mark-holistic-features-WLASL"**
3. Click **Add** on the best match
4. Alternative search terms if needed:
   - "WLASL holistic"
   - "MediaPipe WLASL"
   - "sign language landmarks"

### Step 3: Enable GPU

1. Settings (gear icon) → **Accelerator** → Select **GPU P100** or **GPU T4**
2. Internet: **ON**

### Step 4: Paste & Run Training Script

1. Delete all default cells
2. Copy **ENTIRE** contents of `backend/KAGGLE_TRAINING_V2.py`
3. Paste into Kaggle notebook
4. Click **"Run All"** (or Ctrl+Enter on each cell)

### Step 5: Monitor Training

- You'll see:
  ```
  ✅ Device: cuda
  📂 Found: /kaggle/input/...
  ✅ Loaded XXXX samples from YYY classes
  🚀 TRAINING
  Epoch 1/50 | Train: ... | Val: ...
  ```

- **Good signs:**
  - "Loaded XXXX samples" (not "synthetic data")
  - Validation accuracy improving over epochs
  - Should reach >70% accuracy for good dataset

- **If still fails:**
  - Run `backend/KAGGLE_DATASET_INSPECTOR.py` first
  - Copy output and send back to me

### Step 6: Let It Run

- Close laptop
- Training runs 4-8 hours
- Kaggle saves automatically
- Check back in morning

### Step 7: Download Model

Once training completes:
1. Click **"Output"** tab (bottom right)
2. Download:
   - `best_model.pth`
   - `label_mapping.json`
3. Move files to: `backend/checkpoints/`

---

## Part 2: Test App Locally (While Training Runs)

### Option A: Quick Test with Synthetic Model (Current State)

Since you already downloaded the synthetic model, we can test the architecture:

```bash
cd /home/kathir/Study/Final\ Year\ Project/Code\ by\ Antigravity

# 1. Start backend (Terminal 1)
cd backend
python api/inference_server_wlasl.py

# 2. Start frontend (Terminal 2)
cd frontend
npm run dev
```

**What to expect:**
- ✅ Camera feed works
- ✅ MediaPipe skeleton overlay works
- ✅ Avatar displays
- ⚠️ Sign predictions will be random (synthetic model)
- ✅ Proves the architecture works

### Option B: Test Just MediaPipe Perception

```bash
cd backend
python perception.py --test
```

This tests only the camera + landmark extraction.

### Option C: Full System Test (After Real Model Downloaded)

Once you download the real trained model:

1. Copy files:
   ```bash
   cp ~/Downloads/best_model.pth backend/checkpoints/
   cp ~/Downloads/label_mapping.json backend/checkpoints/
   ```

2. Restart backend server (Ctrl+C then rerun)

3. Test with real signs in front of camera

---

## Troubleshooting

### Issue: "No data loaded" on Kaggle
**Solution:** Run `KAGGLE_DATASET_INSPECTOR.py` first to diagnose

### Issue: Backend won't start
**Solution:** Check if port 8000 busy:
```bash
lsof -i :8000
kill -9 <PID>
```

### Issue: Frontend shows "Connection refused"
**Solution:** Backend not running. Check terminal for errors.

---

## Next Steps After Testing

1. **If training works:** Wait for model, then full end-to-end test
2. **If training fails:** Send me inspector output
3. **If app architecture works:** Move to avatar integration next

**Estimated Timeline:**
- Kaggle training: 4-8 hours (overnight)
- Local testing: 30 minutes
- Full integration: 2-3 days
