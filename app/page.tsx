'use client';
import { useEffect, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useMediaPipe } from '@/hooks/useMediaPipe';
import { useInference } from '@/hooks/useInference';
import { useSpeech } from '@/hooks/useSpeech';

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { processFrame, isProcessing, isLoading: isCameraLoading } = useMediaPipe({
    onLandmarks: (data) => {
      // Draw skeleton
      if (canvasRef.current && videoRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

          // Draw hands
          ctx.fillStyle = '#4ade80'; // Green
          data.leftHand.forEach(pt => {
            ctx.beginPath(); ctx.arc(pt[0] * canvasRef.current!.width, pt[1] * canvasRef.current!.height, 3, 0, 2 * Math.PI); ctx.fill();
          });
          data.rightHand.forEach(pt => {
            ctx.beginPath(); ctx.arc(pt[0] * canvasRef.current!.width, pt[1] * canvasRef.current!.height, 3, 0, 2 * Math.PI); ctx.fill();
          });
        }
      }
      handlePrediction(data);
    }
  });

  const { predict, isLoading: isPredicting } = useInference();
  const { speak, stop, isSpeaking } = useSpeech();

  const [currentSign, setCurrentSign] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [lastSpoken, setLastSpoken] = useState<string>('');
  const lastPredictTime = useRef(0);

  // Initialize Camera
  useEffect(() => {
    let stream: MediaStream | null = null;

    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' }
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play();
            setIsCameraActive(true);

            // Set canvas size to match video
            if (canvasRef.current) {
              canvasRef.current.width = videoRef.current.videoWidth;
              canvasRef.current.height = videoRef.current.videoHeight;
            }
          };
        }
      } catch (err) {
        console.error("Camera failed", err);
      }
    };

    startCamera();

    return () => {
      if (stream) stream.getTracks().forEach(track => track.stop());
    };
  }, []);

  // Frame Loop
  useEffect(() => {
    let animationId: number;

    const loop = () => {
      if (videoRef.current && isCameraActive && !isProcessing) {
        processFrame(videoRef.current, Date.now());
      }
      animationId = requestAnimationFrame(loop);
    };

    if (isCameraActive) loop();
    return () => cancelAnimationFrame(animationId);
  }, [isCameraActive, isProcessing, processFrame]);


  // Prediction Logic
  const handlePrediction = async (landmarks: any) => {
    const now = Date.now();
    // Predict every 500ms max
    if (now - lastPredictTime.current < 500) return;

    // Only predict if hands are visible (simple check)
    const hasHands = landmarks.leftHand.some((p: any) => p[0] !== 0) || landmarks.rightHand.some((p: any) => p[0] !== 0);
    if (!hasHands) return;

    lastPredictTime.current = now;

    const result = await predict(landmarks, 'isl');

    if (result && result.confidence > 0.6) {
      setCurrentSign(result.gloss);
      setConfidence(result.confidence);

      // Auto-speak if new sign (debounce)
      if (result.gloss !== lastSpoken) {
        speak(result.gloss);
        setLastSpoken(result.gloss);
      }
    }
  };

  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-slate-900 via-[#0f172a] to-black text-white overflow-hidden relative selection:bg-indigo-500/30">

      {/* Background Ambience */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden z-0 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-indigo-600/20 rounded-full blur-[120px] animate-pulse" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-600/20 rounded-full blur-[120px] animate-pulse delay-1000" />
      </div>

      <div className="relative z-10 container mx-auto px-4 h-screen flex flex-col items-center justify-center">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="absolute top-8 text-center"
        >
          <h1 className="text-4xl font-thin tracking-[0.3em] bg-clip-text text-transparent bg-gradient-to-r from-indigo-300 to-purple-300">
            SIGNSPEAK
          </h1>
          <p className="text-xs text-indigo-400 mt-1 tracking-widest uppercase opacity-70">
            Real-time Sign Language Translation
          </p>
        </motion.div>

        {/* Main Interface */}
        <div className="flex flex-col items-center gap-8 w-full max-w-4xl">

          {/* Camera Container */}
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="relative group w-full max-w-2xl aspect-[4/3] rounded-3xl overflow-hidden shadow-[0_0_60px_-15px_rgba(79,70,229,0.3)] border border-white/10"
          >
            {/* Camera Feed */}
            <video
              ref={videoRef}
              className="w-full h-full object-cover transform scale-x-[-1]"
              playsInline
              muted
            />

            {/* Skeleton Overlay */}
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full transform scale-x-[-1] opacity-60 pointer-events-none"
            />

            {/* Status Indicators */}
            <div className="absolute top-4 left-4 flex gap-2">
              <div className={`px-3 py-1 rounded-full text-xs font-medium backdrop-blur-md border border-white/10 flex items-center gap-2 ${isCameraActive ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'}`}>
                <div className={`w-2 h-2 rounded-full ${isCameraActive ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
                {isCameraActive ? 'LIVE' : 'OFFLINE'}
              </div>

              {isPredicting && (
                <div className="px-3 py-1 rounded-full text-xs font-medium bg-indigo-500/20 text-indigo-300 backdrop-blur-md border border-white/10 flex items-center gap-2">
                  <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                  PROCESSING
                </div>
              )}
            </div>

          </motion.div>

          {/* Prediction Output */}
          <div className="text-center h-32 flex flex-col items-center justify-center">
            <AnimatePresence mode="wait">
              {currentSign ? (
                <motion.div
                  key={currentSign}
                  initial={{ opacity: 0, y: 20, scale: 0.9 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -20, scale: 0.9 }}
                  className="flex flex-col items-center"
                >
                  <h2 className="text-7xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-indigo-400 to-purple-400 drop-shadow-[0_0_30px_rgba(99,102,241,0.5)]">
                    {currentSign}
                  </h2>
                  <div className="mt-3 flex items-center gap-2 text-indigo-300/60 font-mono text-sm">
                    <span>CONFIDENCE: {Math.round(confidence * 100)}%</span>
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-slate-500 text-xl font-light tracking-wide animate-pulse"
                >
                  Start signing...
                </motion.div>
              )}
            </AnimatePresence>
          </div>

        </div>

        {/* Footer Controls */}
        <div className="absolute bottom-8 flex gap-4">
          {/* Can add mute/unmute buttons here later */}
        </div>
      </div>
    </div>
  );
}
