# Model Checkpoints

This directory contains trained ASL sign language models.

## Files to Download from Kaggle

After running the training script on Kaggle, download these files and place them here:

1. **best_model.pth** (~11 MB)
   - Trained TemporalLSTM model weights
   -Val accuracy: ~16%
   - 25 classes (A-Z alphabet)

2. **label_mapping.json** (~1 KB)
   - Class ID to letter mapping
   - Model metadata

## Usage

The inference server (`backend/api/inference_server_wlasl.py`) automatically loads models from this directory on startup.

```bash
# Start server
cd backend/api
python inference_server_wlasl.py
```

## Model Performance

- Architecture: Bidirectional LSTM
- Parameters: 2.8M
- Input: (30, 150) - 30 frames of 150 features
- Classes: 25 (ASL alphabet, excluding J)
- Validation Accuracy: 16.15%
- Baseline (random): 4%
- Improvement: 4x over random

## Next Steps

1. Download model from Kaggle output
2. Place files here
3. Test locally
4. Deploy to Cloudflare
