# SignSpeak Workflow Guide

## 📋 Complete Flow

### **Phase 1: Training (One-time per new words)**

1. **Collect Data**
   - Open: `yoursite.com/train`
   - Record 40 samples per word
   - Download `word_cache.json`

2. **Upload to Kaggle Dataset**
   - Go to your Kaggle dataset: `kathiravankoffl/manual-isl-cache`
   - Click "New Version"
   - Upload new JSON files
   - Save

3. **Train Model**
   - Open Kaggle training notebook: `kathiravankoffl/training`
   - Click "Run All"
   - Wait for training (5-10 min)
   - Model auto-uploads to `kathiravankoffl/signspeak-model`

---

### **Phase 2: Inference Setup (Every testing session)**

1. **Create Inference Notebook** (One-time)
   - Go to Kaggle → Create new notebook at `kathiravankoffl/backend`
   - Name: "SignSpeak Inference Server"
   - Type: **Interactive Notebook**
   - Accelerator: CPU (or GPU if available)
   - Internet: **ON**
   - Add Model Input:
     - Click "Add Input" → "Models"
     - Search: `kathiravankoffl/signspeak-model`
     - Add (it's your private model)
   - Copy code from `kaggle_inference_notebook.py`
   - Paste into 5 separate cells

2. **Run Inference Server**
   - Open inference notebook
   - Click "Run All" (or run each cell)
   - Wait 30-60 seconds
   - Look for output:
     ```
     ⚠️  COPY THE URL BELOW
     https://abc-xyz-123.trycloudflare.com
     ```
   - **Copy this URL!**

---

### **Phase 3: Frontend Testing**

1. **Open Your Site**
   - Go to deployed site (Cloudflare Pages)

2. **Configure Backend URL**
   - Click Settings icon (top-right)
   - Paste Kaggle tunnel URL
   - Click "Test Connection"
   - If success → Click "Save"

3. **Start Testing!**
   - Click "START DETECTION"
   - Perform signs
   - Words appear in sentence box
   - Click "STOP & SPEAK" → Hears sentence

---

## 🔄 **Daily Workflow** (After initial setup)

```
1. Open Kaggle inference notebook
2. Run all cells (30 sec)
3. Copy tunnel URL
4. Open site → Settings → Paste URL → Save
5. Start signing! 🎯
```

---

## 📝 **Adding New Words**

```
1. Collect 40 samples on /train
2. Upload to Kaggle dataset (new version)
3. Re-run training notebook
4. Restart inference notebook (downloads new model automatically)
5. Frontend gets new vocabulary automatically
```

---

## 🛠️ **Troubleshooting**

**Model Not Found:**
- Check model input is added to inference notebook
- Verify model path: `/kaggle/input/signspeak-model/pytorch/default/1`

**Tunnel Not Working:**
- Make sure Internet is ON in notebook settings
- Restart the last cell if tunnel fails

**Connection Failed in Frontend:**
- URL must start with `https://`
- Copy entire URL including subdomain
- Tunnel URL changes each session

**Low Accuracy:**
- Collect more samples (40+ per word)
- Re-train model
- Check lighting and camera angle

---

## 📁 **File Organization**

```
Kaggle:
├── Dataset: kathiravankoffl/manual-isl-cache
│   └── word_cache.json files
├── Training notebook: kathiravankoffl/training
│   └── Outputs to: kathiravankoffl/signspeak-model
└── Inference notebook: kathiravankoffl/backend
    └── Loads from: kathiravankoffl/signspeak-model

Frontend:
└── Settings UI
    └── Backend URL input
```

---

**Ready to test! 🚀**
