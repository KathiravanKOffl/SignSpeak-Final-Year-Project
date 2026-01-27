# 🚀 Train ASL Model Tonight - Step by Step

## ⏱️ Time Required: 4-6 hours (runs in background)
## 🎯 Target: >90% validation accuracy on WLASL

---

## Step 1: Open Kaggle ✨

1. Go to [kaggle.com](https://www.kaggle.com/)
2. Sign in with your account
3. Click **"Code"** → **"New Notebook"**
4. Name it: `SignSpeak ASL Training`

---

## Step 2: Add Dataset 📦

1. On the right sidebar, click **"+ Add data"**
2. Search: `land-mark-holistic-features-WLASL`
3. Find the dataset with ~200MB size (the one with 2 NPY files)
4. Click **"Add"** button

**Verify it appears in the "Data" panel** →  Should show `land-mark-holistic-featuresfor-wlasl`

---

## Step 3: Enable GPU 🔥

1. Click **Settings** (gear icon on right)
2. **Accelerator** → Select **"GPU P100"** (or GPU T4 if P100 unavailable)
3. **Internet** → Turn **ON**
4. **Persistence** → Session-only (default is fine)

---

## Step 4: Copy Training Script 📋

1. **On your local machine**, open:
   ```
   backend/KAGGLE_TRAINING_WLASL_OPTIMIZED.py
   ```

2. **Select ALL** (Ctrl+A)

3. **Copy** (Ctrl+C)

4. **Back to Kaggle**, in the notebook:
   - Delete any default code cells
   - Paste the entire script (Ctrl+V)

**The script is ~400 lines** - make sure you got everything!

---

## Step 5: Run Training 🚀

1. Click **"Run All"** button (or press Shift+Enter on first cell)

2. **Watch for these messages**:
   ```
   ✅ Device: cuda
   ✅ Loaded data:
      Features: (11980, 30, 75, 2)
      Labels: (11980,)
   ✅ Classes: [actual number, should be ~2000]
   ✅ Data split:
      Train: 10,183 | Val: 1,797
   🚀 TRAINING (Target: >90% Val Acc)
   Epoch   1/150 | Train: ... | Val: ...
   ```

3. **Expected progress**:
   - Epoch 1: ~15-25% validation accuracy
   - Epoch 20: ~50-70%
   - Epoch 50: ~80-90%
   - Epoch 80-100: **>90%** ✅

---

## Step 6: Monitor Training 👀

**Good signs** ✅:
- Validation accuracy steadily increasing
- "💾 Saved!" messages appearing (means new best model)
- Training loss decreasing

**Bad signs** ❌:
- Error messages about CUDA/memory → Reduce batch_size to 64 in CONFIG
- "No data loaded" → Dataset not added correctly, go back to Step 2
- Accuracy stuck at ~5% → Data issue, send me the error

**You can close your laptop!** 
- Kaggle keeps running in their cloud
- Check back in 4-6 hours

---

## Step 7: Download Trained Model 📥

Once training completes (or you see >90% accuracy):

1. Click **"Output"** tab (bottom-right of notebook)

2. You'll see 2 files:
   - `best_model.pth` (~13 MB)
   - `label_mapping.json` (~1 KB)

3. Click the **download icon** next to each file

4. Move them to your project:
   ```bash
   # On your local machine
   mv ~/Downloads/best_model.pth "/home/kathir/Study/Final Year Project/Code by Antigravity/backend/checkpoints/"
   mv ~/Downloads/label_mapping.json "/home/kathir/Study/Final Year Project/Code by Antigravity/backend/checkpoints/"
   ```

---

## Step 8: Test Locally 🧪

Once files are downloaded:

**Terminal 1 - Backend:**
```bash
cd "/home/kathir/Study/Final Year Project/Code by Antigravity/backend"
python api/inference_server_wlasl.py
```

**You should see:**
```
✅ Loaded PyTorch .pth model
✅ Loaded label mapping: XXXX classes
🚀 Uvicorn running on http://0.0.0.0:8000
```

**Terminal 2 - Frontend:**
```bash
cd "/home/kathir/Study/Final Year Project/Code by Antigravity/frontend"
npm run dev
```

**Open browser:** `http://localhost:5173`

**Test it:**
- Allow camera access
- Perform ASL signs in front of camera
- Watch predictions appear!

---

## 🆘 Troubleshooting

### Dataset not found
**Error:** `FileNotFoundError: x_output (1).npy`
**Fix:** Make sure dataset name is exactly `land-mark-holistic-featuresfor-wlasl` (check "Data" panel)

### Out of memory
**Error:** `CUDA out of memory`
**Fix:** In the script, change:
```python
"batch_size": 128,  # Change to 64 or 32
```

### Low accuracy (<50% after 50 epochs)
**Check:**
- Are you using GPU? (Should show `Device: cuda`)
- Is data augmentation working? (Check for "Training" progress bar)

### Training stuck
**If no progress after 5 minutes:**
- Click **"Interrupt"** 
- Check error messages
- Send me screenshot

---

## ✅ Success Criteria

**You're done when:**
1. ✅ Training reaches >90% validation accuracy
2. ✅ `best_model.pth` downloaded
3. ✅ Backend loads model without errors
4. ✅ Frontend shows ASL sign predictions

**Expected timeline:**
- Setup: 10 minutes
- Training: 4-6 hours
- Testing: 30 minutes

**Total: ~6 hours (mostly unattended)**

---

## 📞 Need Help?

If you see any errors, send me:
1. Screenshot of the error
2. Last 10 lines of output
3. Which step you're on

**Now go start the training!** 🚀
