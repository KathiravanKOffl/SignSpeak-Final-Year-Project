'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { Wifi, WifiOff, Eye, EyeOff } from 'lucide-react';
import { motion } from 'framer-motion';
import { useInference } from '@/hooks/useInference';
import { useMediaPipe, type LandmarksData } from '@/hooks/useMediaPipe';

export default function CameraPanel() {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isCameraActive, setIsCameraActive] = useState(false);

    // Cycle state
    const [isActive, setIsActive] = useState(false);
    const [isCountdown, setIsCountdown] = useState(false);
    const [countdownValue, setCountdownValue] = useState(3);
    const [isDetecting, setIsDetecting] = useState(false);
    const [detectionProgress, setDetectionProgress] = useState(0);

    // Output
    const [sentence, setSentence] = useState<string[]>([]);
    const [lastPrediction, setLastPrediction] = useState<string | null>(null);

    // Settings
    const [showSkeleton, setShowSkeleton] = useState(true);

    // Refs
    const detectionStartRef = useRef<number>(0);
    const countdownIntervalRef = useRef<NodeJS.Timeout | null>(null);

    const { predict, isLoading: modelLoading, resetBuffer, error: modelError, isConnected, vocabulary } = useInference();

    const handleLandmarks = useCallback(async (data: LandmarksData) => {
        // Draw skeleton if enabled
        if (showSkeleton) {
            drawSkeleton(data);
        }

        // During detection phase
        if (isDetecting) {
            // Update progress
            const elapsed = Date.now() - detectionStartRef.current;
            const progress = Math.min((elapsed / 2000) * 100, 100);
            setDetectionProgress(progress);

            // Run prediction
            const prediction = await predict(data);

            // Auto-stop after 2 seconds
            if (elapsed >= 2000) {
                setIsDetecting(false);
                setDetectionProgress(100);

                if (prediction) {
                    setSentence(prev => [...prev, prediction]);
                    setLastPrediction(prediction);
                }

                setTimeout(() => setDetectionProgress(0), 300);
            }
        }
    }, [isDetecting, showSkeleton, predict]);

    const { processFrame, isLoading: mediaPipeLoading, error: mediaPipeError } = useMediaPipe({
        onLandmarks: handleLandmarks
    });

    // Countdown logic
    useEffect(() => {
        if (isCountdown && isActive) {
            countdownIntervalRef.current = setInterval(() => {
                setCountdownValue(prev => {
                    if (prev <= 1) {
                        clearInterval(countdownIntervalRef.current!);
                        setIsCountdown(false);
                        // Start detection
                        detectionStartRef.current = Date.now();
                        setIsDetecting(true);
                        resetBuffer();
                        return 3;
                    }
                    return prev - 1;
                });
            }, 1000);

            return () => {
                if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);
            };
        }
    }, [isCountdown, isActive, resetBuffer]);

    // Auto-cycle: After detection ends, start countdown again
    useEffect(() => {
        if (isActive && !isCountdown && !isDetecting) {
            const timer = setTimeout(() => {
                setCountdownValue(3);
                setIsCountdown(true);
            }, 500);
            return () => clearTimeout(timer);
        }
    }, [isActive, isCountdown, isDetecting]);

    // Pose & Hand connections
    const POSE_CONNECTIONS = [
        [11, 12], [11, 23], [12, 24], [23, 24],
        [11, 13], [13, 15], [12, 14], [14, 16],
        [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
        [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
        [24, 26], [26, 28], [28, 30], [28, 32], [30, 32]
    ];

    const HAND_CONNECTIONS = [
        [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
        [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16],
        [0, 17], [17, 18], [18, 19], [19, 20], [5, 9], [9, 13], [13, 17]
    ];

    // Draw skeleton
    const drawSkeleton = (data: LandmarksData) => {
        if (!canvasRef.current) return;
        const ctx = canvasRef.current.getContext('2d');
        if (!ctx) return;

        const width = canvasRef.current.width;
        const height = canvasRef.current.height;
        ctx.clearRect(0, 0, width, height);

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

            pose.forEach((pt, idx) => {
                if (idx >= 15 && idx <= 22) return;
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

        const drawHand = (hand: number[][], color: string) => {
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

            hand.forEach(pt => {
                if (pt && pt[0] !== 0 && pt[1] !== 0) {
                    ctx.fillStyle = color;
                    ctx.beginPath();
                    ctx.arc(pt[0] * width, pt[1] * height, 5, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.fillStyle = '#FFFFFF';
                    ctx.beginPath();
                    ctx.arc(pt[0] * width, pt[1] * height, 3, 0, 2 * Math.PI);
                    ctx.fill();
                }
            });
        };

        if (data.rawPose && data.rawPose.some(pt => pt[0] !== 0)) drawPose(data.rawPose, '#A855F7');
        if (data.rawLeftHand && data.rawLeftHand.some(pt => pt[0] !== 0)) drawHand(data.rawLeftHand, '#10B981');
        if (data.rawRightHand && data.rawRightHand.some(pt => pt[0] !== 0)) drawHand(data.rawRightHand, '#3B82F6');
    };

    // Initialize camera
    useEffect(() => {
        const startCamera = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480, facingMode: 'user' }
                });

                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    videoRef.current.onloadeddata = () => {
                        setIsCameraActive(true);
                        if (canvasRef.current && videoRef.current) {
                            canvasRef.current.width = videoRef.current.videoWidth;
                            canvasRef.current.height = videoRef.current.videoHeight;
                        }
                    };
                }
            } catch (err) {
                console.error('Camera error:', err);
            }
        };

        startCamera();

        return () => {
            if (videoRef.current?.srcObject) {
                const stream = videoRef.current.srcObject as MediaStream;
                stream.getTracks().forEach(track => track.stop());
            }
        };
    }, []);

    // Processing loop
    useEffect(() => {
        if (!isCameraActive || !videoRef.current) return;

        let animationId: number;
        const loop = async () => {
            if (videoRef.current && videoRef.current.readyState === 4) {
                await processFrame(videoRef.current, performance.now());
            }
            animationId = requestAnimationFrame(loop);
        };

        loop();
        return () => { if (animationId) cancelAnimationFrame(animationId); };
    }, [isCameraActive, processFrame]);

    const handleStart = () => {
        setIsActive(true);
        setCountdownValue(3);
        setIsCountdown(true);
        setSentence([]);
        setLastPrediction(null);
    };

    const handleStop = () => {
        setIsActive(false);
        setIsCountdown(false);
        setIsDetecting(false);
        if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);

        // Speak sentence
        if (sentence.length > 0) {
            const text = sentence.join(' ');
            const utterance = new SpeechSynthesisUtterance(text);
            speechSynthesis.speak(utterance);
        }
    };

    const handleClear = () => {
        setSentence([]);
        setLastPrediction(null);
        resetBuffer();
    };

    const isLoading = modelLoading || mediaPipeLoading;
    const error = modelError || mediaPipeError;

    return (
        <div className="h-full bg-white rounded-2xl shadow-sm border border-slate-200 flex flex-col overflow-hidden">
            {/* Header */}
            <div className="p-6 border-b border-slate-200">
                <div className="flex items-center justify-between">
                    <h2 className="text-xl font-bold text-slate-800">Sign Input</h2>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => setShowSkeleton(!showSkeleton)}
                            className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium transition-all ${showSkeleton
                                    ? 'bg-purple-50 text-purple-700 border border-purple-200'
                                    : 'bg-slate-100 text-slate-600 border border-slate-200'
                                }`}
                        >
                            {showSkeleton ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
                            <span>Skeleton</span>
                        </button>

                        <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium ${isConnected
                                ? 'bg-green-50 text-green-700 border border-green-200'
                                : 'bg-amber-50 text-amber-700 border border-amber-200'
                            }`}>
                            {isConnected ? (
                                <>
                                    <Wifi className="w-3 h-3" />
                                    <span>Backend Connected</span>
                                </>
                            ) : (
                                <>
                                    <WifiOff className="w-3 h-3" />
                                    <span>No Backend</span>
                                </>
                            )}
                        </div>
                    </div>
                </div>
                <p className="text-sm text-slate-500 mt-1">3sec countdown → 2sec detection → repeat</p>
            </div>

            {/* Camera View */}
            <div className="flex-1 flex flex-col">
                <div className="flex-1 relative bg-black">
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

                    {/* Countdown Overlay */}
                    {isCountdown && (
                        <div className="absolute inset-0 bg-black/60 flex items-center justify-center z-20">
                            <motion.div
                                key={countdownValue}
                                initial={{ scale: 0.5, opacity: 0 }}
                                animate={{ scale: 1, opacity: 1 }}
                                className="text-9xl font-bold text-white drop-shadow-2xl"
                            >
                                {countdownValue}
                            </motion.div>
                        </div>
                    )}

                    {/* Detection Progress */}
                    {isDetecting && (
                        <>
                            <div className="absolute top-4 right-4 bg-blue-600 text-white px-3 py-1.5 rounded-full text-sm font-bold flex items-center gap-2 z-10 shadow-lg">
                                <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                                DETECTING
                            </div>
                            <div className="absolute bottom-0 left-0 right-0 h-2 bg-slate-200 z-10">
                                <div
                                    className="h-full bg-blue-500 transition-all duration-100 ease-linear"
                                    style={{ width: `${detectionProgress}%` }}
                                />
                            </div>
                        </>
                    )}

                    {/* Last Prediction */}
                    {lastPrediction && !isDetecting && !isCountdown && (
                        <div className="absolute bottom-12 left-1/2 transform -translate-x-1/2 px-6 py-3 bg-green-600 text-white rounded-xl font-bold text-lg shadow-lg">
                            "{lastPrediction}"
                        </div>
                    )}

                    {/* Status */}
                    <div className="absolute top-4 left-4 px-3 py-2 rounded-lg bg-black/60 backdrop-blur-sm">
                        {isLoading ? (
                            <span className="text-yellow-400 text-sm font-medium">⏳ Loading...</span>
                        ) : error ? (
                            <span className="text-red-400 text-sm font-medium">⚠️ Error</span>
                        ) : isActive ? (
                            <span className="text-green-400 text-sm font-medium flex items-center gap-2">
                                <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                                Active
                            </span>
                        ) : (
                            <span className="text-slate-300 text-sm font-medium">● Standby</span>
                        )}
                    </div>
                </div>

                {/* Error */}
                {error && (
                    <div className="bg-red-50 border border-red-200 rounded-xl p-4 mx-4 mt-4 text-red-700 text-sm">
                        {error}
                    </div>
                )}

                {/* Controls */}
                <div className="p-4 bg-slate-50 border-t border-slate-200">
                    <div className="flex gap-3">
                        {!isActive ? (
                            <button
                                onClick={handleStart}
                                disabled={!isCameraActive || !isConnected}
                                className="flex-1 py-3 bg-blue-600 text-white rounded-xl font-bold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-200"
                            >
                                ▶ START
                            </button>
                        ) : (
                            <button
                                onClick={handleStop}
                                className="flex-1 py-3 bg-red-600 text-white rounded-xl font-bold hover:bg-red-700 transition-all shadow-lg shadow-red-200"
                            >
                                ⏸ STOP & SPEAK
                            </button>
                        )}
                        <button
                            onClick={handleClear}
                            disabled={sentence.length === 0}
                            className="px-6 py-3 bg-slate-200 text-slate-700 rounded-xl font-medium hover:bg-slate-300 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                        >
                            Clear
                        </button>
                    </div>
                </div>

                {/* Sentence */}
                <div className="p-4 border-t border-slate-200">
                    <div className="min-h-[60px] bg-slate-50 rounded-xl p-4 border border-slate-200">
                        {sentence.length > 0 ? (
                            <p className="text-lg font-medium text-slate-800">
                                {sentence.join(' ')}
                            </p>
                        ) : (
                            <p className="text-slate-400 italic">Your sentence will appear here...</p>
                        )}
                    </div>
                </div>

                {/* Vocabulary */}
                {vocabulary.length > 0 && (
                    <div className="px-4 pb-4">
                        <p className="text-xs text-slate-500 mb-2 font-medium">Available words:</p>
                        <div className="flex flex-wrap gap-2">
                            {vocabulary.map((word, idx) => (
                                <span
                                    key={idx}
                                    className="px-3 py-1 bg-blue-50 text-blue-700 rounded-full text-xs font-medium border border-blue-200"
                                >
                                    {word}
                                </span>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
