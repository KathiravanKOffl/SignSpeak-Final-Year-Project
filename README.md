# SignSpeak - Real-Time Sign Language Translation

[![Next.js](https://img.shields.io/badge/Next.js-15.1.4-black)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-18.3-blue)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-blue)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A high-performance, real-time sign language translation system supporting ISL (Indian Sign Language) and ASL (American Sign Language). Built with Next.js, FastAPI, and MediaPipe, optimized for cloud-native deployment.

## 🌟 Features

- **Real-Time Translation**: Browser-based MediaPipe for instant landmark extraction.
- **Dual Language Support**: Production-ready ISL and ASL recognition.
- **Cloud-Native Architecture**: Deployed on Cloudflare Pages (Frontend) + Hybrid GPU Backend.
- **Advanced UI/UX**: Professional dashboard with real-time feedback and transcriptions.
- **Multi-Device Sync**: Synchronize camera input, control, and output across devices.
- **AI-Powered Insights**: Integrated with Llama-3 for intelligent text refinement.

## 📁 Project Structure

```
SignSpeak/
├── app/                    # Next.js App Router (UI & API)
│   ├── api/               # Edge Functions (Refinement & Room MGMT)
│   ├── app/               # Main Translation Interface
│   ├── input/             # Remote Camera Input
│   ├── control/           # Remote Controller
│   └── output/            # Live Transcript Display
│
├── components/            # Shared React Components
│   ├── camera/           # MediaPipe Core Module
│   └── transcript/       # Live Transcript UI
│
├── backend/              # Python ML Inference Suite
│   ├── api/             # FastAPI Inference Server
│   ├── model.py         # Sign Recognition Architecture
│   ├── checkpoints/     # Trained Model Weights
│   └── requirements.txt # Backend Dependencies
│
├── docs/                 # Documentation & Architecture
│   ├── DEPLOYMENT.md    # Production Deployment Guide
│   └── architecture/    # Deep-dive Architecture Docs
│
├── public/              # Static Assets & Models
├── package.json        # Frontend Dependencies
└── wrangler.toml       # Cloudflare Configuration
```

## 🚀 Quick Start

### Frontend (Next.js)
```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

### Backend (Inference Server)
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn api.inference_server:app --port 8000
```

## 📦 Deployment

The project is designed for zero-lag edge computing and high-availability GPU inference.

- **Frontend**: Deployed on **Cloudflare Pages** for global edge delivery.
- **Inference**: Hosted on **Hybrid GPU Infrastructure** using FastAPI and secure tunnels.
- **AI Pipelines**: Leverages **Cloudflare Workers AI** (Whisper & Llama) for text and speech processing.

Detailed instructions are available in [DEPLOYMENT.md](docs/DEPLOYMENT.md).

## 🛠️ Tech Stack

- **Frontend**: Next.js 15, React 18, Tailwind CSS, MediaPipe Tasks
- **ML Backend**: PyTorch, FastAPI, CNN-Transformer Hybrid
- **Integration**: Cloudflare Tunnel, Secure WebRTC
- **LLM/STT**: Cloudflare Workers AI (Llama-3, Whisper)

## 💰 Performance & Scalability

| Metric | Target |
|---------|-----------|
| Inference Latency | < 50ms |
| Interface Speed | 60 FPS |
| Cold Start | < 2s |
| Deployment Cost | $0 (Free Tier Optimized) |

---

Built with precision for accessible communication by **Team Kathiravan**.
Licensed under the [MIT License](LICENSE).
