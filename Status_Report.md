# SignSpeak: Project Status Report

**Date:** January 27, 2026
**Project Type:** Final Year Engineering Capstone

---

## 1. Title
**SignSpeak: A Real-Time Bidirectional Multimodal Sign Language Translation System**

---

## 2. Abstract
The "SignSpeak" project aims to bridge the communication gap between the Deaf and Hard-of-Hearing (DHH) community and the hearing majority through a robust, real-time, bidirectional translation framework. Unlike traditional unimodal tools that focus solely on captioning, this system integrates a holistic perception layer (MediaPipe), a bidirectional translation engine (Hybrid CNN-Transformer), and a generative avatar interface into a unified ecosystem. The system is architected to operate within a zero-cost infrastructure using distributed edge computing (Cloudflare Workers) and hybrid cloud inference (Google Colab/Kaggle), ensuring accessibility and scalability.

---

## 3. Project Description
SignSpeak is a comprehensive assistive technology platform designed to facilitate seamless, unstructured communication.
*   **Sign-to-Text/Speech (SLR):** capture continuous sign language gestures (ISL/ASL) via a standard webcam, extracting skeletal landmarks using MediaPipe, and translating them into natural language text using deep learning models.
*   **Speech-to-Sign (SLP):** Converts spoken language into 3D avatar animations, enabling hearing users to communicate back to DHH users in their native sign language.
*   **Key Features:** Real-time latency (<300ms), 543-point holistic tracking (Face, Hands, Pose), and privacy-preserving edge processing.

---

## 4. Existing System
Current solutions for Sign Language Translation face significant limitations:
*   **Glove-Based Systems:** Require expensive, intrusive hardware sensors (flex sensors, accelerometers) that hinder natural movement.
*   **Static Image Classifiers:** Most existing "SignSpeak" implementations use simple CNNs on static images, capable of recognizing only isolated alphabets or numbers, failing at continuous communication.
*   **Unimodal Translation:** Systems typically offer only one-way translation (Sign-to-Text), ignoring the need for the deaf user to receive information back in sign language.
*   **Lack of NMMs:** Existing avatars often lack Non-Manual Markers (facial expressions), resulting in "robotic" and grammatically incomplete signing.

---

## 5. Proposed System
Our proposed SignSpeak system utilizes a state-of-the-art **Hybrid CNN-Transformer** architecture to overcome these limitations.

### Architecture Highlights:
1.  **Perception Layer:** Utilizes **MediaPipe Holistic** to extract 543 skeletal landmarks (Hands, Face, Pose) in real-time on CPU, removing the need for specialized hardware.
2.  **Recognition Engine:**
    *   **Spatial Encoder:** A CNN/MLP extracts geometric features (angles, distances) from each frame.
    *   **Temporal Encoder:** A Transformer mechanism captures long-range dependencies in continuous signing sequences, handling the "co-articulation" of signs.
3.  **Production Engine:** A 3D Avatar (Three.js) driven by NLP-processed glosses, capable of rendering smooth animations and facial expressions.
4.  **Infrastructure:** A distributed microservices architecture using **Cloudflare Tunnels** to securely expose heavy inference backends (like Kaggle/Colab GPUs) to a lightweight web frontend.

---

## 6. Working Progress (Current Status)

**Overall Completion: ~40%**

### ✅ Completed Modules:
*   **Frontend Interface:** Developed a React-based web application with real-time camera integration.
*   **Perception Pipeline:** Successfully integrated MediaPipe Holistic to extract and visualize 543 landmarks in the browser.
*   **Backend Infrastructure:** Set up a FastAPI inference server with Cloudflare Tunneling for secure public access.
*   **Data Pipeline:** Implemented comprehensive data loading scripts for WLASL (ASL) and ISL datasets, including pose-centric normalization.

### 🔄 In Progress:
*   **Model Training:** currently training the Hybrid CNN-Transformer model on Kaggle using the "MuteMotion WLASL" and "Google ASL Signs" datasets.
    *   *Target:* >90% accuracy on isolated signs.
*   **Integration:** Connecting the trained model weights (`best_model.pth`) to the live backend for real-time inference.

### ⏳ Next Steps:
*   **Auditory Engine:** Integrating OpenAI Whisper for Speech-to-Text.
*   **Generative Engine:** connecting the 3D Avatar to strict sign glosses.
*   **System Testing:** End-to-end latency optimization to meet the 300ms target.
