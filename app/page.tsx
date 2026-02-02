'use client';
import { useEffect, useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useMediaPipe, LandmarksData } from '@/hooks/useMediaPipe';
import { useInference } from '@/hooks/useInference';
import { useSpeech } from '@/hooks/useSpeech';

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lastPredictTime = useRef(0);
  const lastSpokenRef = useRef<string>('');

  const [currentSign, setCurrentSign] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);

  const { predict, isLoading: isPredicting } = useInference();
  const { speak } = useSpeech();

  // Handle landmarks and make predictions
  const handleLandmarks = useCallback(async (data: LandmarksData) => {
    // Draw skeleton on canvas
    if (canvasRef.current && data) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

        // Draw hands (green dots)
        ctx.fillStyle = '#4ade80';
        [...data.leftHand, ...data.rightHand].forEach(pt => {
          if (pt[0] !== 0 || pt[1] !== 0) {
            ctx.beginPath();
            ctx.arc(pt[0] * canvasRef.current!.width, pt[1] * canvasRef.current!.height, 4, 0, 2 * Math.PI);
            ctx.fill();
          }
        });
      }
    }

    // Throttle predictions to every 600ms
    const now = Date.now();
    if (now - lastPredictTime.current < 600) return;

    // Check if hands are visible
    const hasHands = data.leftHand.some(p => p[0] !== 0) || data.rightHand.some(p => p[0] !== 0);
    if (!hasHands) return;

    lastPredictTime.current = now;

    try {
      const result = await predict(data, 'isl');

      // Show prediction if we have a result (low threshold for now)
      if (result && result.gloss) {
        console.log('[App] Showing prediction:', result.gloss, 'confidence:', result.confidence);
        setCurrentSign(result.gloss);
        setConfidence(result.confidence);

        // Speak if different from last spoken
        if (result.gloss !== lastSpokenRef.current) {
          speak(result.gloss);
          lastSpokenRef.current = result.gloss;
        }
      }
    } catch (err) {
      console.error('[Prediction Error]', err);
    }
  }, [predict, speak]);

  // Initialize MediaPipe with landmarks handler
  const { processFrame, isLoading: isMediaPipeLoading } = useMediaPipe({
    onLandmarks: handleLandmarks,
  });

  // Initialize Camera
  useEffect(() => {
    let stream: MediaStream | null = null;
    let mounted = true;

    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' }
        });

        if (!mounted || !videoRef.current) return;

        videoRef.current.srcObject = stream;

        videoRef.current.onloadedmetadata = () => {
          if (!videoRef.current || !mounted) return;

          videoRef.current.play().then(() => {
            if (!mounted) return;

            // Set canvas size
            if (canvasRef.current && videoRef.current) {
              canvasRef.current.width = videoRef.current.videoWidth || 640;
              canvasRef.current.height = videoRef.current.videoHeight || 480;
            }

            setIsCameraActive(true);
            setIsInitialized(true);
          }).catch(console.error);
        };
      } catch (err) {
        console.error("Camera initialization failed:", err);
      }
    };

    startCamera();

    return () => {
      mounted = false;
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // Frame processing loop - only runs when camera is active and MediaPipe is ready
  useEffect(() => {
    if (!isCameraActive || isMediaPipeLoading || !isInitialized) return;

    let animationId: number;
    let lastFrameTime = 0;
    const targetFPS = 15; // Limit to 15 FPS to prevent overload
    const frameInterval = 1000 / targetFPS;

    const loop = (timestamp: number) => {
      if (timestamp - lastFrameTime >= frameInterval) {
        if (videoRef.current && videoRef.current.readyState >= 2) {
          processFrame(videoRef.current, timestamp);
        }
        lastFrameTime = timestamp;
      }
      animationId = requestAnimationFrame(loop);
    };

    animationId = requestAnimationFrame(loop);

    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [isCameraActive, isMediaPipeLoading, isInitialized, processFrame]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-indigo-950 to-black text-white overflow-hidden relative">

      {/* Background Glow */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-indigo-600/15 rounded-full blur-[100px]" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-purple-600/15 rounded-full blur-[100px]" />
      </div>

      <div className="relative z-10 container mx-auto px-4 min-h-screen flex flex-col items-center justify-center py-8">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-light tracking-[0.2em] bg-clip-text text-transparent bg-gradient-to-r from-indigo-300 to-purple-300">
            SIGNSPEAK
          </h1>
          <p className="text-sm text-indigo-400 mt-2 opacity-70">
            Real-time Indian Sign Language Translation
          </p>
        </motion.div>

        {/* Camera Container */}
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.4 }}
          className="relative w-full max-w-2xl aspect-video rounded-2xl overflow-hidden shadow-2xl shadow-indigo-500/20 border border-white/10"
        >
          {/* Video Feed */}
          <video
            ref={videoRef}
            className="w-full h-full object-cover transform scale-x-[-1]"
            playsInline
            muted
            autoPlay
          />

          {/* Skeleton Canvas Overlay */}
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full transform scale-x-[-1] pointer-events-none"
          />

          {/* Status Badge */}
          <div className="absolute top-4 left-4">
            <div className={`px-3 py-1.5 rounded-full text-xs font-medium backdrop-blur-md border flex items-center gap-2 ${isCameraActive
                ? 'bg-green-500/20 text-green-300 border-green-500/30'
                : 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30'
              }`}>
              <div className={`w-2 h-2 rounded-full ${isCameraActive ? 'bg-green-400 animate-pulse' : 'bg-yellow-400'}`} />
              {isCameraActive ? 'LIVE' : 'INITIALIZING...'}
            </div>
          </div>

          {/* Processing Indicator */}
          {isPredicting && (
            <div className="absolute top-4 right-4">
              <div className="px-3 py-1.5 rounded-full text-xs font-medium bg-indigo-500/20 text-indigo-300 backdrop-blur-md border border-indigo-500/30 flex items-center gap-2">
                <div className="w-3 h-3 border-2 border-indigo-300 border-t-transparent rounded-full animate-spin" />
                ANALYZING
              </div>
            </div>
          )}

        </motion.div>

        {/* Prediction Output */}
        <div className="mt-10 text-center min-h-[120px] flex flex-col items-center justify-center">
          <AnimatePresence mode="wait">
            {currentSign ? (
              <motion.div
                key={currentSign}
                initial={{ opacity: 0, y: 15, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -15, scale: 0.95 }}
                transition={{ duration: 0.2 }}
                className="flex flex-col items-center"
              >
                <h2 className="text-6xl md:text-7xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-indigo-400 to-purple-400">
                  {currentSign}
                </h2>
                <div className="mt-4 text-indigo-300/60 font-mono text-sm">
                  CONFIDENCE: {Math.round(confidence * 100)}%
                </div>
              </motion.div>
            ) : (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-slate-500 text-xl font-light"
              >
                {isMediaPipeLoading ? 'Loading AI models...' : 'Start signing to translate'}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Footer */}
        <div className="absolute bottom-4 text-center text-xs text-slate-600">
          Powered by MediaPipe & ISL Recognition Model
        </div>
      </div>
    </div>
  );
}
