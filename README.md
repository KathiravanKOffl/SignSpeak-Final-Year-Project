# SignSpeak: Bidirectional Real-Time Sign Language Translation System

A comprehensive real-time, bidirectional sign language translation system enabling seamless communication between signers (using Indian Sign Language - ISL and American Sign Language - ASL) and non-signers (using spoken/written English/Tamil/Hindi).

## 🎯 Key Features

- **Dual Language Support**: Full support for both ISL and ASL
- **Real-time Translation**: Sign-to-Text/Speech and Speech-to-Sign with <1.2s latency
- **Zero-Cost Infrastructure**: 100% deployed on Cloudflare Pages + Workers AI + Google Colab free tiers
- **Privacy-Preserving**: All video processing at the edge, only skeletal landmarks transmitted
- **Dual Operational Modes**:
  - **Unified Mode**: Single-device split-screen experience
  - **Multi-Device Mode**: Distributed setup via shareable room links
- **Advanced Features**: Federated learning, emotion-aware avatars, regional dialects, fingerspelling

## 🏗️ Architecture

### Deployment
- **Single Cloudflare Pages Application** with two modes
- Cloudflare Durable Objects for room state management
- Google Colab T4 GPU for custom model inference
- Cloudflare Workers AI for ASR and NLP

### Tech Stack
- **Frontend**: Next.js 14 + TypeScript + Tailwind CSS + Three.js
- **Computer Vision**: MediaPipe Holistic (TensorFlow.js in browser)
- **Backend ML**: Python 3.10 + PyTorch + MediaPipe
- **Cloud**: Cloudflare Pages + Workers + Durable Objects + Colab

## 📁 Project Structure

```
signspeak/
├── signspeak-web/          # Next.js application
│   ├── app/                # Next.js App Router pages
│   ├── components/         # React components
│   ├── hooks/              # Custom React hooks
│   ├── lib/                # Utility libraries
│   └── workers/            # Cloudflare Workers (Durable Objects)
├── backend/                # Python ML backend
│   ├── model.py            # CNN-Transformer architecture
│   ├── train.py            # Training pipeline
│   ├── data_loader.py      # Dataset processing
│   └── colab_inference.ipynb  # Colab deployment notebook
├── python-utils/           # Python utilities
│   └── perception.py       # MediaPipe landmark extraction
├── docs/                   # Documentation
└── tests/                  # Test suites
```

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.10+
- Conda (recommended)
- Cloudflare account (free tier)
- Google account for Colab

### Setup

1. **Clone and install dependencies**:
```bash
# Install Next.js dependencies
cd signspeak-web
npm install

# Set up Python environment
conda create -n signspeak python=3.10
conda activate signspeak
pip install -r ../requirements.txt
```

2. **Configure environment variables**:
```bash
# Create .env.local in signspeak-web/
cp .env.example .env.local
# Add your Cloudflare API tokens and Colab tunnel URL
```

3. **Run development server**:
```bash
# Frontend
npm run dev

# Python backend (for local testing)
python ../backend/perception.py
```

4. **Deploy to Cloudflare Pages**:
```bash
npm install -g wrangler
wrangler login
npx @cloudflare/next-on-pages
npx wrangler pages deploy
```

## 📖 Documentation

- [Implementation Plan](docs/implementation_plan.md)
- [Architecture Overview](docs/architecture.md)
- [API Documentation](docs/api.md)
- [Deployment Guide](docs/deployment.md)

## 🎓 Academic Context

This project is part of a Final Year Project exploring real-time sign language translation using cutting-edge AI and zero-cost cloud infrastructure.

**Key Research Areas**:
- Hybrid CNN-Transformer architectures for sign recognition
- Self-supervised learning with VideoMAE
- Privacy-preserving federated learning
- Zero-cost cloud deployment strategies

## 📊 Performance Targets

- Sign Recognition Accuracy: >95% (benchmark: 96.8%)
- End-to-End Latency: <1.2s
- Avatar Rendering: 30+ FPS
- Zero monetary cost (100% free tier)

## 🤝 Contributing

This is an academic project, but contributions and feedback are welcome!

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- MediaPipe team for landmark detection
- Cloudflare for free-tier infrastructure
- WLASL, INCLUDE, CISLR dataset creators
- DHH community for feedback and testing

---

**Status**: 🚧 Under Development

Last Updated: January 26, 2026
