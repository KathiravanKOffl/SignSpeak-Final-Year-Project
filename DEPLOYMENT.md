# Frontend Deployment to Cloudflare Pages

## 🚀 **Quick Deploy**

### **Option 1: GitHub Integration (Recommended)**

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add Kaggle backend integration"
   git push origin main
   ```

2. **Cloudflare auto-deploys!**
   - Your existing integration will build and deploy automatically
   - No manual steps needed

---

### **Option 2: Manual Deploy**

```bash
npm run deploy
```

---

## ⚙️ **First-Time Setup (After Deployment)**

1. **Start Kaggle Inference Notebook:**
   - Go to: `kaggle.com/notebooks/kathiravankoffl/backend`
   - Run all cells
   - Wait for cloudflared tunnel URL

2. **Configure Frontend:**
   - Visit your deployed site
   - Click ⚙️ (Settings icon in header)
   - Paste Kaggle tunnel URL
   - Click "Test Connection"
   - Click "Save & Close"

3. **Start Using:**
   - Camera will activate
   - Click "Start Detection"
   - Sign words
   - See predictions!

---

## 🔄 **Updating Model**

When you add new words:

1. **Train on Kaggle** (`kathiravankoffl/training`)
2. **Upload new model** to `kathiravankoffl/signspeak-model`
3. **Restart inference notebook**
4. **No frontend changes needed!** Vocabulary auto-syncs

---

## 📝 **Dependencies Added**

- ✅ `lucide-react` - Icons
- ❌ Removed `onnxruntime-web` (no longer needed)

---

## 🆘 **Troubleshooting**

**"No Backend" Warning:**
- Open Settings (⚙️)
- Configure tunnel URL
- Test connection

**Connection Test Fails:**
- Ensure Kaggle notebook is running
- Check tunnel URL is correct (starts with `https://`)
- Verify Internet is ON in Kaggle settings

**Predictions Not Working:**
- Check "Backend Connected" shows green
- Ensure model is trained with vocabulary
- Try restarting detection

---

## ✅ **Ready to Deploy!**

All code changes are complete. Just push to GitHub and configuration is done! 🎉
