# Google Colab Inference Setup

## 🚀 **Quick Start**

### **1. Open in Colab**
- Go to: https://colab.research.google.com
- File → Upload notebook
- Upload: `colab_inference_notebook.py` (rename to `.ipynb` or copy cells)

### **2. Prepare Files**
Download from Kaggle training output:
- `sign_model.pth` (trained model)
- `vocabulary.json` (word list)

### **3. Run Notebook**
1. **Cell 1:** Installs dependencies (~30 seconds)
2. **Cell 2:** Upload files when prompted
   - Click "Choose Files"
   - Select `sign_model.pth` and `vocabulary.json`
   - Upload both
3. **Cell 3:** Loads model
4. **Cell 4:** Sets up inference function
5. **Cell 5:** Creates FastAPI server
6. **Cell 6:** Starts tunnel
   - Copy the cloudflared URL
   - Paste in frontend settings

### **4. Configure Frontend**
- Open your deployed site
- Click ⚙️ (Settings)
- Paste Colab tunnel URL
- Test connection
- Save

---

## ✅ **Advantages over Kaggle**

✅ More stable for long-running cells  
✅ Better free tier GPU access  
✅ Simpler file upload (drag & drop)  
✅ Faster startup time  
✅ Less likely to timeout  

---

## 🔄 **Updating Model**

When you train new words:

1. **Train on Kaggle** (training notebook)
2. **Download** `sign_model.pth` + `vocabulary.json`
3. **Re-run Colab notebook** (upload new files)
4. **Copy new tunnel URL** to frontend

---

## ⚠️ **Important Notes**

- **Keep Colab tab open** - minimize to stay connected
- **Tunnel URL changes** each time you restart
- **Free tier limits:** ~12 hours per session
- **Update frontend URL** after each restart

---

## 📋 **Troubleshooting**

**Upload fails:**
- Files must be named exactly: `sign_model.pth`, `vocabulary.json`
- Try uploading one at a time

**Tunnel stops:**
- Normal after some hours
- Just re-run Cell 6
- Copy new URL to frontend

**Out of memory:**
- Use CPU runtime (GPU not needed for inference)
- Restart runtime and try again

---

## 🎯 **Ready!**

Colab is now your inference backend. Works great with the existing frontend! 🚀
