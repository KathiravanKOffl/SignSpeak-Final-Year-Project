# SignSpeak - Real-Time Sign Language Translation

[![Next.js](https://img.shields.io/badge/Next.js-15.1.4-black)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-18.3-blue)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-blue)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A zero-cost, real-time sign language translation system supporting ISL (Indian Sign Language) and ASL (American Sign Language). Built with Next.js, Python, MediaPipe, and deployed on Cloudflare Pages + Google Colab.

## 🌟 Features

- **Real-Time Translation**: Browser-based MediaPipe for instant landmark extraction
- **Dual Language Support**: ISL (263 signs) and ASL (2,000 signs)
- **Zero Cost Deployment**: Cloudflare Pages (frontend) + Google Colab (ML backend)
- **Multi-Device Mode**: Separate camera, control, and output screens
- **Privacy-First**: All processing on-device and edge
- **Hybrid Architecture**: WebRTC for camera + FastAPI for ML inference

## 📁 Project Structure

```
SignSpeak-Final-Year-Project/
├── app/                    # Next.js App Router
│   ├── api/               # API routes
│   │   ├── predict/       # Sign recognition endpoint
│   │   ├── transcribe/    # Speech-to-text (Whisper)
│   │   ├── translate/     # Text-to-gloss (Llama-3)
│   │   └── room/          # Multi-device room management
│   ├── app/               # Main translation page
│   ├── input/             # Camera input page
│   ├── control/           # Camera control page
│   ├── output/            # Translation output page
│   ├── layout.tsx         # Root layout
│   ├── page.tsx           # Landing page
│   └── globals.css        # Global styles
│
├── components/            # React components
│   ├── camera/           # Camera module with MediaPipe
│   └── transcript/       # Transcript panel
│
├── hooks/                # Custom React hooks
│   └── useMediaPipe.ts  # MediaPipe hook
│
├── stores/               # State management (Zustand)
│   └── appStore.ts      # Global app state
│
├── backend/              # Python ML backend
│   ├── model.py         # CNN-Transformer model
│   ├── train.py         # Training pipeline
│   ├── api/             # FastAPI server
│   │   └── inference_server.py
│   ├── requirements.txt # Python dependencies
│   └── colab_deployment.ipynb  # Colab deployment
│
├── docs/                 # Documentation
│   ├── DEPLOYMENT.md    # Deployment guide
│   └── CLOUDFLARE_ARCHITECTURE.md
│
├── public/              # Static assets
├── python-utils/        # Perception utilities
│   └── perception.py   # MediaPipe utilities
│
├── package.json        # Node.js dependencies
├── tsconfig.json       # TypeScript config
├── tailwind.config.ts  # Tailwind CSS config
├── next.config.ts      # Next.js config
├── wrangler.toml       # Cloudflare Pages config
└── README.md          # This file
```

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.10+ (for backend development)
- Google Colab account (for ML backend deployment)
- Cloudflare account (for frontend deployment)

### Local Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Open http://localhost:3000
```

### Build for Production

```bash
# Build Next.js app
npm run build

# Start production server
npm start
```

## 📦 Deployment

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed deployment instructions.

### Quick Deploy to Cloudflare Pages

1. Push code to GitHub
2. Connect repository to Cloudflare Pages
3. Configure build settings:
   - Build command: `npm install && npm run build`
   - Output directory: `.next`
4. Set environment variables
5. Deploy!

### Deploy Backend to Google Colab

1. Open `backend/colab_deployment.ipynb` in Colab
2. Enable T4 GPU
3. Run all cells
4. Copy tunnel URL
5. Update Cloudflare Pages environment variable

## 🛠️ Tech Stack

### Frontend
- **Framework**: Next.js 15 (App Router)
- **UI**: React 18, TypeScript, Tailwind CSS
- **3D**: Three.js, React Three Fiber
- **Computer Vision**: MediaPipe Tasks Vision
- **State**: Zustand
- **Deployment**: Cloudflare Pages

### Backend
- **ML Framework**: PyTorch
- **Model**: CNN-Transformer Hybrid
- **API**: FastAPI + Uvicorn
- **Deployment**: Google Colab (T4 GPU)
- **Tunnel**: Cloudflare Tunnel

### AI Services
- **Speech Recognition**: Cloudflare Workers AI (Whisper)
- **Text Processing**: Cloudflare Workers AI (Llama-3)

## 💰 Cost Breakdown

| Service | Free Tier | Usage | Cost |
|---------|-----------|-------|------|
| Cloudflare Pages | 500 builds/month | ~10/month | $0 |
| Cloudflare Workers AI | 10,000 req/day | ~100/day | $0 |
| Google Colab | 12 hours/session | As needed | $0 |
| **TOTAL** | | | **$0/month** |

## 🎯 Performance

- **Inference Latency**: <50ms (sign recognition)
- **Camera FPS**: 30fps (MediaPipe)
- **Model Size**: ~50MB (quantized)
- **Page Load**: <2s (Cloudflare CDN)

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

## 📧 Contact

For questions or support, open an issue on GitHub.

---

Built with ❤️ for accessible communication
