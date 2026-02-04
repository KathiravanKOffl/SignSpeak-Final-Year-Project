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
    // MediaPipe pose connections
    const POSE_CONNECTIONS = [
        // Torso
        [11, 12], [11, 23], [12, 24], [23, 24],
        // Left arm
        [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
        // Right arm  
        [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
        // Face
        [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
        // Left leg
        [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
        // Right leg
        [24, 26], [26, 28], [28, 30], [28, 32], [30, 32]
    ];

    // MediaPipe hand connections
    const HAND_CONNECTIONS = [
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15], [15, 16],
        [0, 17], [17, 18], [18, 19], [19, 20],
        [5, 9], [9, 13], [13, 17]
    ];

    // Draw skeleton using canvas API
    const drawSkeleton = (data: LandmarksData) => {
        if (!canvasRef.current) return;

        const ctx = canvasRef.current.getContext('2d');
        if (!ctx) return;

        const width = canvasRef.current.width;
        const height = canvasRef.current.height;

        ctx.clearRect(0, 0, width, height);

        // Helper to draw pose body
        const drawPose = (pose: number[][], color: string) => {
            ctx.strokeStyle = color;
            ctx.lineWidth = 4;
            ctx.lineCap = 'round';

            POSE_CONNECTIONS.forEach(([start, end]) => {
                const startPt = pose[start];
                const endPt = pose[end];

                if (startPt && endPt && startPt[0] !== 0 && startPt[1] !== 0 && endPt[0] !== 0 && endPt[1] !== 0) {
                    ctx.beginPath();
                    ctx.moveTo(startPt[0] * width, startPt[1] * height);
                    ctx.lineTo(endPt[0] * width, endPt[1] * height);
                    ctx.stroke();
                }
            });

            // Draw joints
            pose.forEach(pt => {
                if (pt && pt[0] !== 0 && pt[1] !== 0) {
                    ctx.fillStyle = color;
                    ctx.beginPath();
                    ctx.arc(pt[0] * width, pt[1] * height, 6, 0, 2 * Math.PI);
                    ctx.fill();

                    ctx.fillStyle = '#FFFFFF';
                    ctx.beginPath();
                    ctx.arc(pt[0] * width, pt[1] * height, 3, 0, 2 * Math.PI);
                    ctx.fill();
                }
            });
        };

        // Helper to draw hand with connections
        const drawHand = (hand: number[][], color: string, fillColor: string) => {
            // Draw connections (bones)
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.lineCap = 'round';

            HAND_CONNECTIONS.forEach(([start, end]) => {
                const startPt = hand[start];
                const endPt = hand[end];

                if (startPt && endPt && startPt[0] !== 0 && startPt[1] !== 0 && endPt[0] !== 0 && endPt[1] !== 0) {
                    ctx.beginPath();
                    ctx.moveTo(startPt[0] * width, startPt[1] * height);
                    ctx.lineTo(endPt[0] * width, endPt[1] * height);
                    ctx.stroke();
                }
            });

            // Draw joints (landmarks)
            hand.forEach(pt => {
                if (pt && pt[0] !== 0 && pt[1] !== 0) {
                    // Outer circle (border)
                    ctx.fillStyle = color;
                    ctx.beginPath();
                    ctx.arc(pt[0] * width, pt[1] * height, 5, 0, 2 * Math.PI);
                    ctx.fill();

                    // Inner circle (fill)
                    ctx.fillStyle = fillColor;
                    ctx.beginPath();
                    ctx.arc(pt[0] * width, pt[1] * height, 3, 0, 2 * Math.PI);
                    ctx.fill();
                }
            });
        };

        // Helper to draw face landmarks
        const drawFace = (face: number[][], color: string) => {
            face.forEach(pt => {
                if (pt && pt[0] !== 0 && pt[1] !== 0) {
                    ctx.fillStyle = color;
                    ctx.beginPath();
                    ctx.arc(pt[0] * width, pt[1] * height, 2, 0, 2 * Math.PI);
                    ctx.fill();
                }
            });
        };

        // Use RAW coordinates (0-1 range) for accurate screen drawing
        // Draw body pose (purple/magenta)
        if (data.rawPose && data.rawPose.some(pt => pt[0] !== 0)) {
            drawPose(data.rawPose, '#A855F7');
        }

        // Draw face (yellow)
        if (data.rawFace && data.rawFace.some(pt => pt[0] !== 0)) {
            drawFace(data.rawFace, '#EAB308');
        }

        // Draw left hand (green)
        if (data.rawLeftHand && data.rawLeftHand.some(pt => pt[0] !== 0)) {
            drawHand(data.rawLeftHand, '#10B981', '#FFFFFF');
        }

        // Draw right hand (blue)
        if (data.rawRightHand && data.rawRightHand.some(pt => pt[0] !== 0)) {
            drawHand(data.rawRightHand, '#3B82F6', '#FFFFFF');
        }
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
            {/* Header */}
            <div className="mb-6 text-center">
                <h1 className="text-3xl font-bold text-slate-800 mb-2">Training System</h1>
                <p className="text-slate-500 text-sm">Collect samples for word recognition model</p>
            </div>

            {/* 2-Column Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 max-w-7xl mx-auto">

                {/* Left Column: Camera */}
                <div className="space-y-4">
                    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
                        <div className="relative bg-slate-100" style={{ aspectRatio: '4/3' }}>
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
                                <div className="absolute top-4 right-4 bg-red-500 text-white px-4 py-2 rounded-full text-sm font-bold flex items-center gap-2 z-10">
                                    <div className="w-3 h-3 bg-white rounded-full animate-pulse" />
                                    RECORDING ({Math.ceil((100 - recordingProgress) / 50)}s)
                                </div>
                            )}

                            {/* Recording progress */}
                            {isRecording && (
                                <div className="absolute bottom-0 left-0 right-0 h-2 bg-slate-200 z-10">
                                    <motion.div
                                        className="h-full bg-red-500"
                                        initial={{ width: 0 }}
                                        animate={{ width: `${recordingProgress}%` }}
                                    />
                                </div>
                            )}

                            {/* Status overlay */}
                            {(isModelLoading || !isCameraActive) && (
                                <div className="absolute inset-0 bg-white/90 flex items-center justify-center z-20">
                                    <div className="text-center">
                                        <div className="w-12 h-12 border-4 border-slate-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-4" />
                                        <p className="text-slate-600 font-medium">{status}</p>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Info */}
                    <div className="p-4 bg-blue-50 border border-blue-100 rounded-xl">
                        <p className="text-xs text-blue-700">
                            <strong>Note:</strong> The skeleton is just visual feedback. The system captures landmark data regardless of how the drawing looks.
                        </p>
                    </div>
                </div>

                {/* Right Column: Controls */}
                <div className="space-y-6">

                    {/* Word Display */}
                    <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl shadow-lg p-8 text-white">
                        <p className="text-sm uppercase tracking-wider mb-2 opacity-90">Current Word</p>
                        <h2 className="text-6xl font-bold mb-6">{currentWord}</h2>
                        <div className="flex justify-between items-end">
                            <div>
                                <p className="text-sm opacity-90">Sample Progress</p>
                                <p className="text-3xl font-bold">{samples.length}/{samplesPerWord}</p>
                            </div>
                            <div className="text-right">
                                <p className="text-sm opacity-90">Word Progress</p>
                                <p className="text-xl font-bold">{currentWordIndex + 1}/{vocabulary.length}</p>
                            </div>
                        </div>
                    </div>

                    {/* Progress Bar */}
                    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
                        <div className="flex justify-between text-sm text-slate-600 mb-2">
                            <span>Collection Progress</span>
                            <span>{Math.round((samples.length / samplesPerWord) * 100)}%</span>
                        </div>
                        <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                            <motion.div
                                className="h-full bg-green-500"
                                initial={{ width: 0 }}
                                animate={{ width: `${(samples.length / samplesPerWord) * 100}%` }}
                                transition={{ duration: 0.3 }}
                            />
                        </div>
                    </div>

                    {/* Action Buttons */}
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
                                        className="w-full py-5 bg-blue-600 text-white rounded-xl font-bold text-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-600/30"
                                    >
                                        {isRecording ? '⏺ Recording...' : '▶ START RECORDING'}
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
                                        className="py-5 bg-slate-200 text-slate-700 rounded-xl font-bold text-lg hover:bg-slate-300 transition-all"
                                    >
                                        ↺ RETRY
                                    </button>
                                    <button
                                        onClick={handleConfirm}
                                        className="py-5 bg-green-600 text-white rounded-xl font-bold text-lg hover:bg-green-700 transition-all shadow-lg shadow-green-600/30"
                                    >
                                        ✓ CONFIRM
                                    </button>
                                </motion.div>
                            )}
                        </AnimatePresence>

                        {/* Secondary Actions */}
                        <div className="grid grid-cols-2 gap-4">
                            <button
                                onClick={resetWord}
                                className="py-3 bg-white border-2 border-slate-200 text-slate-600 rounded-xl font-medium hover:bg-slate-50 transition-all"
                            >
                                Reset Word
                            </button>
                            <button
                                onClick={nextWord}
                                disabled={samples.length < samplesPerWord}
                                className="py-3 bg-white border-2 border-slate-200 text-slate-600 rounded-xl font-medium hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                            >
                                Skip Word →
                            </button>
                        </div>
                    </div>

                    {/* Status */}
                    <div className="bg-white rounded-xl border border-slate-200 p-4">
                        <div className="flex items-center gap-3">
                            <div className={`w-3 h-3 rounded-full ${isCameraActive ? 'bg-green-500 animate-pulse' : 'bg-slate-300'}`} />
                            <span className="text-sm font-medium text-slate-700">{status}</span>
                        </div>
                    </div>

                </div>

            </div>
        </main>
    );
}
