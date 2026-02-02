# SignSpeak - Final Year Project

**Indian Sign Language Recognition System**

> **Status:** ✅ **Model Training Complete - 76.06% Accuracy Achieved!**

---

## 🎯 Project Overview

Real-time Indian Sign Language (ISL) recognition system using MediaPipe landmarks and Transformer-based deep learning.

### **Current Achievements:**
- ✅ **123-class model trained:** 76.06% validation accuracy
- ✅ **Model size:** 29.4 MB (deployment-ready)
- ✅ **Training time:** 8.5 minutes on Tesla P100
- ✅ **Dataset:** 2,235 samples, fully validated
- ✅ **Zero data leakage:** Stratified 80/20 split verified

---

## 📁 Project Structure

### **Training Notebooks** (Kaggle-ready)

| # | Notebook | Purpose | Status | Size |
|---|----------|---------|--------|------|
| **1** | `1-isl-123-training.ipynb` | **Train 123-class model** | ✅ **Complete (76.06%)** | 34 KB |
| 2 | `2-isl-123-cache-extraction.ipynb` | Extract MediaPipe landmarks | ✅ Done | 58 KB |
| 3 | `3-utility-create-labels.ipynb` | Generate file→label mapping | ℹ️ Utility | 13 KB |

### **Documentation**

| File | Purpose |
|------|---------|
| `README.md` | This file - Project overview |
| `TRAINING-GUIDE.md` | Detailed training guide |
| `QUICK-REFERENCE.md` | Quick lookup & FAQ |
| `docs/DEPLOYMENT.md` | Deployment instructions |
| `docs/CLOUDFLARE_ARCHITECTURE.md` | Cloud architecture |

---

## 🚀 Quick Start - Model Training

### **Use Notebook 1: `1-isl-123-training.ipynb`**

**What it does:**
1. Auto-generates `file_to_label.json` from INCLUDE dataset
2. Filters to 123 target classes
3. Creates stratified train/val split (80/20)
4. Trains Transformer model with augmentation
5. Achieves **76.06% validation accuracy**

**Inputs needed:**
- INCLUDE dataset (source videos)
- isl-123-cache (pre-extracted landmarks)

**Outputs:**
- `best_isl_123.pth` (29.4 MB trained model)
- `file_to_label.json` (sample→class mapping)
- `history.json` (training metrics)
- `results.png` (loss/accuracy plots)

**Runtime:** ~8-9 minutes on Tesla P100 GPU

---

## 🧠 Model Architecture

### **Configuration:**
```python
{
    'input_dim': 408,          # 136 landmarks × 3 coords
    'hidden_dim': 384,         # Optimal size
    'num_layers': 4,           # Transformer layers
    'num_heads': 8,            # Attention heads
    'num_classes': 123,        # ISL signs
    'dropout': 0.4,            # Regularization
    'parameters': 7.35M,       # Model size
    'file_size': 29.4MB        # Disk usage
}
```

### **Training Details:**
- **Optimizer:** AdamW (lr=1e-4, wd=0.02)
- **Scheduler:** ReduceLROnPlateau
- **Loss:** CrossEntropy + Label Smoothing (0.15)
- **Augmentation:** 7 techniques (time warp, noise, rotation, scaling, masking, temporal shift, mixup)
- **Batch size:** 32
- **Epochs:** 150 (early stop at 142)

---

## 📊 Performance Metrics

### **Final Results:**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Validation Accuracy** | **76.06%** | 73-78% | ✅ **Achieved!** |
| Train Accuracy | 95.58% | - | ✅ |
| Training Time | 8.51 min | <10 min | ✅ |
| Model Size | 29.4 MB | <50 MB | ✅ |
| Classes Covered | 123/123 | 123 | ✅ |
| Data Leakage | 0 overlap | 0 | ✅ |
| Best Epoch | 142/150 | - | ✅ |

### **Training Progress:**

```
Epoch 50:  45.64% val acc
Epoch 100: 72.26% val acc
Epoch 116: 75.17% val acc  🎯 First target hit!
Epoch 142: 76.06% val acc  ✅ Best!
```

### **Model Evolution:**

| Version | Model Size | Val Acc | Issue |
|---------|------------|---------|-------|
| v1 | 13M params | 72.04% | Below target, overfitting |
| v2 | 2.5M params | 59.96% | Underfitting |
| **v3 (Final)** | **7.35M** | **76.06%** | ✅ **Perfect!** |

---

## 🎯 Dataset Information

### **123-Class Dataset:**
- **Total samples:** 2,235
- **Classes:** 123 common ISL signs
- **Avg samples/class:** 18.2
- **Min samples/class:** 5
- **Max samples/class:** 25
- **Train/Val split:** 1,788 / 447 (80/20 stratified)
- **Format:** Pre-extracted MediaPipe landmarks (.npy)

### **Example Classes:**
adult, alright, animal, baby, bad, bank, bicycle, big, bird, black, boy, girl, hello, man, pink, water, yes, no, etc.

---

## 🔧 Technologies Used

### **ML/AI:**
- PyTorch 2.8.0
- MediaPipe (landmark extraction)
- Transformer architecture
- Data augmentation (7 techniques)

### **Deployment:**
- Cloudflare Workers/Pages
- Edge computing
- WebRTC for video

### **Development:**
- Python 3.12
- JavaScript/TypeScript
- Next.js (UI)

---

## 📈 Development Timeline

| Date | Milestone |
|------|-----------|
| Jan 2026 | Data collection & preprocessing |
| Feb 1, 2026 | Cache extraction complete |
| Feb 2, 2026 | ✅ **Training complete (76.06%)** |
| Feb 2026 | Deployment & testing |
| Mar 2026 | Final presentation |

---

## 🎓 Academic Context

- **Project:** Final Year B.Tech Project
- **Domain:** Computer Vision + Accessibility
- **Focus:** Real-time ISL recognition
- **Impact:** Enable communication for deaf community

---

## 👤 Author

**Kathiravan K**  
Final Year B.Tech Student  
2026

---

## 📝 Notes

### **Production Deployment:**
- Model is ready (29.4 MB)
- Achieves target accuracy (76.06%)
- Suitable for edge devices
- No further training needed

### **Dataset Scope:**
- Focused on 123 most common ISL signs
- Balanced for practical use cases
- Sufficient for real-world communication

---

**Last Updated:** February 2, 2026  
**Model Version:** v3 (Final)  
**Status:** ✅ Production Ready
