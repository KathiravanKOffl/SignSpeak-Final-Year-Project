'use client';

import { useEffect, useRef, useState } from 'react';
import { useMediaPipe, LandmarksData } from '@/hooks/useMediaPipe';
import { useInference } from '@/hooks/useInference';
import { useChatStore } from '@/stores/chatStore';

export default function CameraPanel() {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isCameraActive, setIsCameraActive] = useState(false);
    const { addMessage } = useChatStore();
    const { predict, isLoading: isPredicting, getBufferSize, BUFFER_SIZE } = useInference();
    const lastPredictionRef = useRef<string | null>(null);

    // Handle landmarks from MediaPipe
    const handleLandmarks = async (data: LandmarksData) => {
        // 1. Draw Skeleton
        if (canvasRef.current && videoRef.current) {
            const ctx = canvasRef.current.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

                // Draw hands (Professional Green)
                ctx.fillStyle = '#10B981'; // Emerald-500
                [...data.leftHand, ...data.rightHand].forEach(pt => {
                    // Ensure visibility (some landmarks might be zero if not detected)
                    if (pt[0] !== 0 && pt[1] !== 0) {
                        ctx.beginPath();
                        ctx.arc(pt[0] * canvasRef.current!.width, pt[1] * canvasRef.current!.height, 3, 0, 2 * Math.PI);
                        ctx.fill();
                    }
                });
            }
        }

        // 2. Inference Logic
        const hasHands = data.leftHand.some(p => p[0] !== 0) || data.rightHand.some(p => p[0] !== 0);
        if (!hasHands) return;

        try {
            const result = await predict(data, 'isl');
            if (result && result.gloss) {
                // Debounce: only add if different from last recent prediction or sufficient time passed
                // For now, strict change detection to avoid spam
                if (result.gloss !== lastPredictionRef.current && result.confidence > 0.6) {
                    console.log(`[Signer] Predicted: ${result.gloss} (${result.confidence})`);
                    addMessage('signer', result.gloss);
                    lastPredictionRef.current = result.gloss;

                    // Reset debounce after 2 seconds
                    setTimeout(() => { lastPredictionRef.current = null; }, 2000);
                }
            }
        } catch (err) {
            console.error("Prediction error:", err);
        }
    };

    // Initialize MediaPipe
    const { processFrame, isLoading: isModelLoading } = useMediaPipe({
        onLandmarks: handleLandmarks,
    });

    // Start Camera
    useEffect(() => {
        let stream: MediaStream | null = null;

        const startCamera = async () => {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480, facingMode: 'user' }
                });
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    videoRef.current.onloadeddata = () => {
                        setIsCameraActive(true);
                        if (videoRef.current && canvasRef.current) {
                            canvasRef.current.width = videoRef.current.videoWidth;
                            canvasRef.current.height = videoRef.current.videoHeight;
                        }
                    };
                }
            } catch (err) {
                console.error("Camera failed:", err);
            }
        };

        startCamera();

        return () => {
            stream?.getTracks().forEach(t => t.stop());
        };
    }, []);

    // Frame Loop
    useEffect(() => {
        if (!isCameraActive || isModelLoading) return;

        let frameId: number;
        const loop = (timestamp: number) => {
            if (videoRef.current && videoRef.current.readyState >= 2) {
                processFrame(videoRef.current, timestamp);
            }
            frameId = requestAnimationFrame(loop);
        };
        frameId = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(frameId);
    }, [isCameraActive, isModelLoading, processFrame]);

    return (
        <div className="flex flex-col h-full bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
            {/* Header */}
            <div className="px-6 py-4 border-b border-slate-100 bg-slate-50 flex justify-between items-center">
                <h2 className="font-semibold text-slate-700 flex items-center gap-2">
                    <div className={`w-2.5 h-2.5 rounded-full ${isCameraActive ? 'bg-emerald-500 animate-pulse' : 'bg-amber-400'}`} />
                    Signer Input
                </h2>
                <div className="text-xs font-medium text-slate-400 uppercase tracking-wider">User 1</div>
            </div>

            {/* Video Container */}
            <div className="flex-1 relative bg-slate-100 overflow-hidden flex items-center justify-center group">
                <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="absolute inset-0 w-full h-full object-cover transform scale-x-[-1]"
                />
                <canvas
                    ref={canvasRef}
                    className="absolute inset-0 w-full h-full transform scale-x-[-1] pointer-events-none opacity-60"
                />

                {/* Loading State */}
                {(isModelLoading || !isCameraActive) && (
                    <div className="absolute inset-0 bg-white/80 backdrop-blur-sm z-10 flex flex-col items-center justify-center">
                        <div className="w-8 h-8 border-4 border-slate-200 border-t-blue-500 rounded-full animate-spin mb-4" />
                        <p className="text-slate-500 text-sm font-medium">Initializing Vision Engine...</p>
                    </div>
                )}

                {/* Inference Indicator */}
                {isPredicting && (
                    <div className="absolute top-4 right-4 bg-black/60 backdrop-blur-md text-white text-[10px] px-2 py-1 rounded-full font-medium">
                        PROCESSING
                    </div>
                )}

                {/* Controls Overlay (Bottom) */}
                <div className="absolute bottom-6 left-6 right-6 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                    <div className="bg-white/90 backdrop-blur-md rounded-xl p-4 shadow-lg border border-white/20">
                        <div className="flex justify-between text-xs text-slate-500 mb-2">
                            <span>Confidence Threshold</span>
                            <span>{(getBufferSize() / BUFFER_SIZE * 100).toFixed(0)}%</span>
                        </div>
                        <div className="h-1.5 bg-slate-200 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-blue-500 transition-all duration-300 ease-out"
                                style={{ width: `${(getBufferSize() / BUFFER_SIZE) * 100}%` }}
                            />
                        </div>
                    </div>
                </div>
            </div>

            {/* Footer info */}
            <div className="px-6 py-3 bg-white border-t border-slate-100 text-[11px] text-slate-400 flex justify-between">
                <span>ISL Model v3 (123 Classes)</span>
                <span>Running on MediaPipe</span>
            </div>
        </div>
    );
}
