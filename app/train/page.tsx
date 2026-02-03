'use client';

import { useEffect, useRef, useState } from 'react';
import { useMediaPipe, LandmarksData } from '@/hooks/useMediaPipe';
import { useTrainingStore } from '@/stores/trainingStore';
import { resampleToN, flattenFrame } from '@/utils/frameNormalization';
import { motion, AnimatePresence } from 'framer-motion';

export default function TrainPage() {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const frameBufferRef = useRef<LandmarksData[]>([]);
    const recordingStartRef = useRef<number>(0);

    const [isCameraActive, setIsCameraActive] = useState(false);
    const [status, setStatus] = useState<string>('Initializing...');

    const {
        currentWord,
        currentWordIndex,
        vocabulary,
        samples,
        samplesPerWord,
        isRecording,
        recordingProgress,
        showPreview,
        previewData,
        startRecording,
        stopRecording,
        setRecordingProgress,
        confirmSample,
        retrySample,
        resetWord,
        nextWord
    } = useTrainingStore();

    // MediaPipe initialization
    const { processFrame, isLoading: isModelLoading } = useMediaPipe({
        onLandmarks: (data) => {
            // Draw skeleton
            drawSkeleton(data);

            // Collect frames during recording
            if (isRecording) {
                frameBufferRef.current.push(data);

                // Calculate progress (2 seconds = 100%)
                const elapsed = Date.now() - recordingStartRef.current;
                const progress = Math.min((elapsed / 2000) * 100, 100);
                setRecordingProgress(progress);

                // Auto-stop after 2 seconds
                if (elapsed >= 2000) {
                    handleAutoStop();
                }
            }
        }
    });

    // Initialize camera
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
                        setStatus('Ready');

                        if (canvasRef.current && videoRef.current) {
                            canvasRef.current.width = videoRef.current.videoWidth;
                            canvasRef.current.height = videoRef.current.videoHeight;
                        }
                    };
                }
            } catch (err) {
                console.error('Camera error:', err);
                setStatus('Camera failed');
            }
        };

        startCamera();

        return () => {
            stream?.getTracks().forEach(t => t.stop());
        };
    }, []);

    // Frame processing loop
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

    // MediaPipe hand connections
    const HAND_CONNECTIONS = [
        [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
        [0, 5], [5, 6], [6, 7], [7, 8],       // Index
        [0, 9], [9, 10], [10, 11], [11, 12],  // Middle
        [0, 13], [13, 14], [14, 15], [15, 16],// Ring
        [0, 17], [17, 18], [18, 19], [19, 20],// Pinky
        [5, 9], [9, 13], [13, 17]             // Palm
    ];

    // Draw skeleton on canvas
    const drawSkeleton = (data: LandmarksData) => {
        if (!canvasRef.current) return;

        const ctx = canvasRef.current.getContext('2d');
        if (!ctx) return;

        const width = canvasRef.current.width;
        const height = canvasRef.current.height;

        ctx.clearRect(0, 0, width, height);

        // Helper to draw hand
        const drawHand = (hand: number[][], color: string) => {
            // Draw connections (lines)
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            HAND_CONNECTIONS.forEach(([start, end]) => {
                const startPt = hand[start];
                const endPt = hand[end];

                if (startPt[0] !== 0 && startPt[1] !== 0 && endPt[0] !== 0 && endPt[1] !== 0) {
                    ctx.beginPath();
                    ctx.moveTo(startPt[0] * width, startPt[1] * height);
                    ctx.lineTo(endPt[0] * width, endPt[1] * height);
                    ctx.stroke();
                }
            });

            // Draw joints (circles)
            ctx.fillStyle = color;
            hand.forEach(pt => {
                if (pt[0] !== 0 && pt[1] !== 0) {
                    ctx.beginPath();
                    ctx.arc(pt[0] * width, pt[1] * height, 5, 0, 2 * Math.PI);
                    ctx.fill();
                }
            });
        };

        // Draw both hands (different colors)
        drawHand(data.leftHand, '#10B981');   // Left = Green
        drawHand(data.rightHand, '#3B82F6');  // Right = Blue
    };

    // Start recording
    const handleStart = () => {
        frameBufferRef.current = [];
        recordingStartRef.current = Date.now();
        startRecording();
        setStatus('Recording...');
    };

    // Auto-stop after 2 seconds
    const handleAutoStop = () => {
        stopRecording();

        // Normalize to 32 frames
        const normalized = resampleToN(frameBufferRef.current, 32);

        // Show preview
        useTrainingStore.setState({
            showPreview: true,
            previewData: normalized
        });

        setStatus('Review your sign');
    };

    // Confirm sample
    const handleConfirm = () => {
        if (previewData) {
            confirmSample(previewData);
            setStatus(`Saved! (${samples.length + 1}/${samplesPerWord})`);
        }
    };

    // Retry sample
    const handleRetry = () => {
        retrySample();
        setStatus('Ready');
    };

    return (
        <main className="min-h-screen bg-[#F8F9FA] p-6">
            <div className="max-w-4xl mx-auto">

                {/* Header */}
                <div className="mb-8 text-center">
                    <h1 className="text-3xl font-bold text-slate-800 mb-2">Training System</h1>
                    <p className="text-slate-500 text-sm">Collect samples for word recognition model</p>
                </div>

                {/* Word Progress */}
                <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 mb-6">
                    <div className="flex justify-between items-center mb-4">
                        <div>
                            <p className="text-sm text-slate-500 uppercase tracking-wider mb-1">Current Word</p>
                            <h2 className="text-4xl font-bold text-blue-600">{currentWord}</h2>
                        </div>
                        <div className="text-right">
                            <p className="text-sm text-slate-500 uppercase tracking-wider mb-1">Progress</p>
                            <p className="text-3xl font-bold text-slate-700">
                                {samples.length}/{samplesPerWord}
                            </p>
                            <p className="text-xs text-slate-400 mt-1">
                                Word {currentWordIndex + 1} of {vocabulary.length}
                            </p>
                        </div>
                    </div>

                    {/* Progress bar */}
                    <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                        <motion.div
                            className="h-full bg-blue-600"
                            initial={{ width: 0 }}
                            animate={{ width: `${(samples.length / samplesPerWord) * 100}%` }}
                            transition={{ duration: 0.3 }}
                        />
                    </div>
                </div>

                {/* Camera View */}
                <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden mb-6">
                    <div className="relative aspect-video bg-slate-100">
                        <video
                            ref={videoRef}
                            autoPlay
                            playsInline
                            muted
                            className="absolute inset-0 w-full h-full object-cover transform scale-x-[-1]"
                        />
                        <canvas
                            ref={canvasRef}
                            className="absolute inset-0 w-full h-full transform scale-x-[-1] pointer-events-none"
                        />

                        {/* Recording indicator */}
                        {isRecording && (
                            <div className="absolute top-4 right-4 bg-red-500 text-white px-4 py-2 rounded-full text-sm font-bold flex items-center gap-2">
                                <div className="w-3 h-3 bg-white rounded-full animate-pulse" />
                                RECORDING ({Math.ceil((100 - recordingProgress) / 50)}s)
                            </div>
                        )}

                        {/* Recording progress */}
                        {isRecording && (
                            <div className="absolute bottom-0 left-0 right-0 h-1 bg-slate-200">
                                <motion.div
                                    className="h-full bg-red-500"
                                    initial={{ width: 0 }}
                                    animate={{ width: `${recordingProgress}%` }}
                                />
                            </div>
                        )}

                        {/* Status overlay */}
                        {(isModelLoading || !isCameraActive) && (
                            <div className="absolute inset-0 bg-white/90 flex items-center justify-center">
                                <div className="text-center">
                                    <div className="w-12 h-12 border-4 border-slate-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-4" />
                                    <p className="text-slate-600 font-medium">{status}</p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Controls */}
                <div className="space-y-4">
                    <AnimatePresence mode="wait">
                        {!showPreview ? (
                            <motion.div
                                key="record"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                            >
                                <button
                                    onClick={handleStart}
                                    disabled={isRecording || !isCameraActive || isModelLoading}
                                    className="w-full py-4 bg-blue-600 text-white rounded-xl font-semibold text-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-600/20"
                                >
                                    {isRecording ? 'Recording...' : 'START RECORDING'}
                                </button>
                            </motion.div>
                        ) : (
                            <motion.div
                                key="review"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                className="grid grid-cols-2 gap-4"
                            >
                                <button
                                    onClick={handleRetry}
                                    className="py-4 bg-slate-200 text-slate-700 rounded-xl font-semibold text-lg hover:bg-slate-300 transition-all"
                                >
                                    RETRY
                                </button>
                                <button
                                    onClick={handleConfirm}
                                    className="py-4 bg-green-600 text-white rounded-xl font-semibold text-lg hover:bg-green-700 transition-all shadow-lg shadow-green-600/20"
                                >
                                    CONFIRM
                                </button>
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {/* Additional controls */}
                    <div className="flex gap-4">
                        <button
                            onClick={resetWord}
                            className="flex-1 py-3 bg-slate-100 text-slate-600 rounded-xl font-medium hover:bg-slate-200 transition-all"
                        >
                            Reset Word
                        </button>
                        <button
                            onClick={nextWord}
                            disabled={samples.length < samplesPerWord}
                            className="flex-1 py-3 bg-slate-100 text-slate-600 rounded-xl font-medium hover:bg-slate-200 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                        >
                            Skip Word →
                        </button>
                    </div>
                </div>

                {/* Info */}
                <div className="mt-6 p-4 bg-blue-50 border border-blue-100 rounded-xl">
                    <p className="text-sm text-blue-700">
                        <strong>How it works:</strong> Press START → Sign the word → System auto-stops after 2 seconds → Review → Confirm or Retry → Repeat {samplesPerWord} times → Download cache
                    </p>
                </div>

            </div>
        </main>
    );
}
