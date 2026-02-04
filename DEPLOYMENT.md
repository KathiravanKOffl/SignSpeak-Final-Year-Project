# Cloudflare Deployment Guide

## Pre-deployment Checklist ✅

- [x] ONNX model exported (`sign_model.onnx`)
- [x] Vocabulary file ready (`vocabulary.json`)
- [x] Files placed in `/public/models/`
- [x] `package.json` updated with `onnxruntime-web`
- [x] `useInference.ts` configured for ONNX
- [x] `CameraPanel.tsx` implements sentence construction
- [x] All imports fixed

## Deployment Steps

### 1. Install Dependencies
```bash
npm install
```

### 2. Build for Cloudflare Pages
```bash
npm run pages:build
```

### 3. Deploy
```bash
npm run deploy
```

Or deploy via Cloudflare Dashboard:
1. Go to Cloudflare Pages
2. Connect GitHub repository
3. Build settings:
   - Build command: `npm run pages:build`
   - Build output: `.vercel/output/static`
4. Deploy!

## After Deployment

### Testing:
1. Open deployed URL
2. Allow camera access
3. Click **START DETECTION**
4. Perform signs: "I", "you", "he"
5. Click **STOP & SPEAK** → Should speak sentence
6. Click **CLEAR** → Reset

### Adding More Words:
1. Collect data on `/train` page
2. Upload to Kaggle dataset (new version)
3. Re-run training notebook
4. Download new `sign_model.onnx`
5. Replace in `/public/models/`
6. Update vocabulary in `useInference.ts` line 14
7. Re-deploy!

## Important Notes

- Model runs **entirely in browser** (no backend needed)
- Works on any modern device
- First load may take 2-3 seconds (model loading)
- Requires HTTPS for camera access (Cloudflare provides this)

## File Locations

```
/public/models/
  ├── sign_model.onnx      (trained model)
  └── vocabulary.json       (word list)

/hooks/
  └── useInference.ts       (ONNX inference logic)

/components/
  └── CameraPanel.tsx       (main UI with sentence construction)
```

Ready for deployment! 🚀
