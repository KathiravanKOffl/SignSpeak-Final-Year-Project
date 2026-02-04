'use client';

import { useEffect, useRef, useState } from 'react';
import { useMediaPipe, LandmarksData } from '@/hooks/useMediaPipe';
import { useTrainingStore, FULL_VOCABULARY } from '@/stores/trainingStore';
import { resampleToN } from '@/utils/frameNormalization';
import { motion, AnimatePresence } from 'framer-motion';

// Tier definitions
const TIERS = {
    'Tier 1: Grammar Core': FULL_VOCABULARY.slice(0, 10),
    'Tier 2: Common Verbs': FULL_VOCABULARY.slice(10, 25),
    'Tier 3: Essential Nouns': FULL_VOCABULARY.slice(25, 40),
    'Tier 4: Modifiers': FULL_VOCABULARY.slice(40, 50)
};

export default function TrainPage() {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const frameBufferRef = useRef<LandmarksData[]>([]);
    const recordingStartRef = useRef<number>(0);
    const countdownIntervalRef = useRef<NodeJS.Timeout | null>(null);
    const isProcessingStopRef = useRef(false); // Guard against double-stop

    const [isCameraActive, setIsCameraActive] = useState(false);
    const [status, setStatus] = useState<string>('Initializing...');
    const [showWordChanger, setShowWordChanger] = useState(false);
    const [selectedTier, setSelectedTier] = useState<string | null>(null);
    const [isStarted, setIsStarted] = useState(false); // Track if user clicked START

    const {
        currentWord,
        currentWordIndex,
        vocabulary,
        samples,
        samplesPerWord,
        isRecording,
        recordingProgress,
        isCountdown,
        countdownValue,
        isPaused,
        setCurrentWord,
        startCountdown,
        startRecording,
        stopRecording,
        setRecordingProgress,
        setCountdownValue,
        setIsCountdown,
        pauseTraining,
        resumeTraining,
        confirmSample,
        retrySample,
        resetWord
    } = useTrainingStore();

    // MediaPipe initialization
    const { processFrame, isLoading: isModelLoading } = useMediaPipe({
        onLandmarks: (data) => {
            // Draw skeleton
            drawSkeleton(data);

            // Collect frames during recording
            if (isRecording && !isProcessingStopRef.current) {
                frameBufferRef.current.push(data);

                // Calculate progress (2 seconds = 100%)
                const elapsed = Date.now() - recordingStartRef.current;
                const progress = Math.min((elapsed / 2000) * 100, 100);
                setRecordingProgress(progress);

                // Auto-stop after 2 seconds
                if (elapsed >= 2000) {
                    isProcessingStopRef.current = true; // Prevent double-trigger
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
                        setStatus('Ready to start');

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

    // Countdown logic
    useEffect(() => {
        if (isCountdown && !isPaused) {
            countdownIntervalRef.current = setInterval(() => {
                const current = useTrainingStore.getState().countdownValue;
                if (current > 1) {
                    setCountdownValue(current - 1);
                } else {
                    setIsCountdown(false);
                    clearInterval(countdownIntervalRef.current!);
                    handleStart();
                }
            }, 1000);
            return () => {
                if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);
            };
        }
    }, [isCountdown, isPaused]);

    // Auto-mode: immediately start next countdown after recording (only if user clicked START)
    useEffect(() => {
        if (isStarted && !isCountdown && !isRecording && samples.length < samplesPerWord && !isPaused) {
            const timer = setTimeout(() => {
                startCountdown();
            }, 500);
            return () => clearTimeout(timer);
        }
    }, [isStarted, isCountdown, isRecording, samples.length, samplesPerWord, isPaused]);

    const handleStart = () => {
        frameBufferRef.current = [];
        recordingStartRef.current = Date.now();
        isProcessingStopRef.current = false; // Reset guard
        setIsStarted(true); // Enable auto mode
        startRecording();
        setStatus('Recording...');
    };

    const handleAutoStop = () => {
        if (!isProcessingStopRef.current) return; // Already processed

        stopRecording();
        setRecordingProgress(100); // Force 100% display

        const normalized = resampleToN(frameBufferRef.current, 32);

        // Auto-confirm - immediately save and continue  
        confirmSample(normalized);
        setStatus(`Sample ${samples.length + 1}/${samplesPerWord} saved`);

        // Reset guard for next recording
        setTimeout(() => {
            isProcessingStopRef.current = false;
        }, 100);
    };

    const handleCancel = () => {
        setIsStarted(false); // Stop auto mode
        setIsCountdown(false);
        if (countdownIntervalRef.current) {
            clearInterval(countdownIntervalRef.current);
        }
        setStatus('Ready to start');
    };

    const handleWordChange = (word: string) => {
        const wordIndex = vocabulary.indexOf(word);
        if (wordIndex !== -1) {
            setCurrentWord(word, wordIndex);
            setShowWordChanger(false);
            setSelectedTier(null);
            setStatus(`Switched to: ${word}`);
        }
    };

    return (
        <>
            <main className="min-h-screen bg-[#F8F9FA] flex flex-col">
                <div className="p-4 sm:p-5 border-b bg-white">
                    <div className="max-w-7xl mx-auto">
                        <h1 className="text-2xl sm:text-3xl font-bold text-slate-800">Training - Auto Mode</h1>
                        <p className="text-slate-500 text-sm mt-1">{samplesPerWord} samples per word • Auto-capture enabled</p>
                    </div>
                </div>

                <div className="flex-1 flex flex-col lg:grid lg:grid-cols-2 gap-3 sm:gap-4 p-3 sm:p-4 max-w-7xl mx-auto w-full">

                    {/* Camera */}
                    <div className="relative bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden" style={{ aspectRatio: '4/3' }}>
                        <video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 w-full h-full object-cover transform scale-x-[-1]" />
                        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full transform scale-x-[-1] pointer-events-none" />

                        {/* Countdown */}
                        {isCountdown && (
                            <div className="absolute inset-0 bg-black/60 flex items-center justify-center z-20">
                                <motion.div
                                    key={countdownValue}
                                    initial={{ scale: 0.5, opacity: 0 }}
                                    animate={{ scale: 1, opacity: 1 }}
                                    className="text-7xl sm:text-9xl font-bold text-white"
                                >
                                    {countdownValue}
                                </motion.div>
                            </div>
                        )}

                        {/* Recording indicator */}
                        {isRecording && (
                            <>
                                <div className="absolute top-3 sm:top-4 right-3 sm:right-4 bg-red-500 text-white px-2 sm:px-3 py-1.5 sm:py-2 rounded-full text-xs sm:text-sm font-bold flex items-center gap-2 z-10">
                                    <div className="w-2 sm:w-3 h-2 sm:h-3 bg-white rounded-full animate-pulse" />
                                    REC ({Math.ceil((100 - recordingProgress) / 50)}s)
                                </div>
                                <div className="absolute bottom-0 left-0 right-0 h-1.5 sm:h-2 bg-slate-200 z-10">
                                    <div
                                        className="h-full bg-red-500 transition-all duration-100 ease-linear"
                                        style={{ width: `${recordingProgress}%` }}
                                    />
                                </div>
                            </>
                        )}

                        {(isModelLoading || !isCameraActive) && (
                            <div className="absolute inset-0 bg-white/90 flex items-center justify-center z-20">
                                <div className="text-center px-4">
                                    <div className="w-10 h-10 sm:w-12 sm:h-12 border-4 border-slate-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-3 sm:mb-4" />
                                    <p className="text-slate-600 font-medium text-sm sm:text-base">{status}</p>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Controls */}
                    <div className="flex flex-col gap-3 sm:gap-4">
                        <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl shadow-lg p-4 sm:p-6 text-white">
                            <p className="text-xs uppercase tracking-wider mb-1 opacity-90">Current Word</p>
                            <h2 className="text-4xl sm:text-5xl font-bold mb-3 sm:mb-4">{currentWord}</h2>
                            <div className="flex justify-between text-xs sm:text-sm">
                                <span>{samples.length}/{samplesPerWord} samples</span>
                                <span>Word {currentWordIndex + 1}/{vocabulary.length}</span>
                            </div>
                        </div>

                        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-3 sm:p-4">
                            <div className="flex justify-between text-xs text-slate-600 mb-2">
                                <span>Progress</span>
                                <span>{Math.round((samples.length / samplesPerWord) * 100)}%</span>
                            </div>
                            <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                                <motion.div className="h-full bg-green-500" animate={{ width: `${(samples.length / samplesPerWord) * 100}%` }} />
                            </div>
                        </div>

                        {/* Controls */}
                        <div className="grid grid-cols-2 gap-2 sm:gap-3">
                            {!isRecording && !isCountdown && !isStarted && (
                                <button
                                    onClick={() => {
                                        setIsStarted(true);
                                        startCountdown();
                                    }}
                                    disabled={!isCameraActive || isModelLoading}
                                    className="col-span-2 py-4 sm:py-5 bg-blue-600 text-white rounded-xl font-bold text-lg sm:text-xl hover:bg-blue-700 disabled:opacity-50 shadow-lg"
                                >
                                    ▶ START
                                </button>
                            )}

                            {!isRecording && !isCountdown && isStarted && (
                                <div className="col-span-2 py-4 sm:py-5 bg-slate-100 text-slate-600 rounded-xl font-bold text-lg sm:text-xl flex items-center justify-center gap-2">
                                    <div className="w-5 h-5 border-3 border-slate-300 border-t-blue-600 rounded-full animate-spin" />
                                    Processing...
                                </div>
                            )}

                            {isCountdown && (
                                <>
                                    <button
                                        onClick={isPaused ? resumeTraining : pauseTraining}
                                        className="py-3 sm:py-4 bg-yellow-500 text-white rounded-xl font-bold text-sm sm:text-base"
                                    >
                                        {isPaused ? '▶ RESUME' : '⏸ PAUSE'}
                                    </button>
                                    {isPaused ? (
                                        <button
                                            onClick={() => {
                                                retrySample();
                                                handleCancel();
                                            }}
                                            className="py-3 sm:py-4 bg-red-500 text-white rounded-xl font-bold text-sm sm:text-base"
                                        >
                                            ↺ RETRY
                                        </button>
                                    ) : (
                                        <button
                                            onClick={handleCancel}
                                            className="py-3 sm:py-4 bg-red-500 text-white rounded-xl font-bold text-sm sm:text-base"
                                        >
                                            ✕ CANCEL
                                        </button>
                                    )}
                                </>
                            )}
                        </div>

                        {/* Secondary Actions */}
                        <div className="grid grid-cols-2 gap-2 sm:gap-3">
                            <button
                                onClick={() => setShowWordChanger(true)}
                                className="py-2.5 sm:py-3 bg-white border-2 border-blue-200 text-blue-600 rounded-xl font-medium hover:bg-blue-50 text-sm sm:text-base"
                            >
                                Change Word
                            </button>
                            <button
                                onClick={resetWord}
                                className="py-2.5 sm:py-3 bg-white border-2 border-slate-200 text-slate-600 rounded-xl font-medium hover:bg-slate-50 text-sm sm:text-base"
                            >
                                Reset Word
                            </button>
                        </div>

                        {/* Status */}
                        <div className="bg-white rounded-xl border border-slate-200 p-2.5 sm:p-3 text-center">
                            <div className="flex items-center justify-center gap-2">
                                <div className={`w-2 h-2 rounded-full ${isCameraActive ? 'bg-green-500 animate-pulse' : 'bg-slate-300'}`} />
                                <span className="text-xs sm:text-sm font-medium text-slate-700">{status}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </main>

            {/* Word Changer Modal */}
            <AnimatePresence>
                {showWordChanger && (
                    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={() => setShowWordChanger(false)}>
                        <motion.div
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.9, opacity: 0 }}
                            onClick={(e) => e.stopPropagation()}
                            className="bg-white rounded-2xl shadow-2xl max-w-md w-full max-h-[80vh] overflow-y-auto"
                        >
                            <div className="sticky top-0 bg-white border-b p-4 sm:p-6">
                                <h2 className="text-xl sm:text-2xl font-bold text-slate-800">Change Word</h2>
                                <p className="text-xs sm:text-sm text-slate-500 mt-1">Select a tier, then choose a word</p>
                            </div>

                            <div className="p-4 sm:p-6">
                                {!selectedTier ? (
                                    <div className="space-y-2 sm:space-y-3">
                                        {Object.entries(TIERS).map(([name, words]) => (
                                            <button
                                                key={name}
                                                onClick={() => setSelectedTier(name)}
                                                className="w-full p-3 sm:p-4 rounded-xl border-2 border-slate-200 hover:border-blue-300 hover:bg-blue-50 text-left transition-all"
                                            >
                                                <div className="font-bold text-slate-800 text-sm sm:text-base">{name}</div>
                                                <div className="text-xs sm:text-sm text-slate-500 mt-1">{words.length} words</div>
                                            </button>
                                        ))}
                                    </div>
                                ) : (
                                    <>
                                        <button
                                            onClick={() => setSelectedTier(null)}
                                            className="text-blue-600 text-sm font-medium mb-3 sm:mb-4 flex items-center gap-1"
                                        >
                                            ← Back to Tiers
                                        </button>
                                        <div className="grid grid-cols-3 gap-2">
                                            {TIERS[selectedTier as keyof typeof TIERS].map(word => (
                                                <button
                                                    key={word}
                                                    onClick={() => handleWordChange(word)}
                                                    className={`py-2 sm:py-2.5 px-2 sm:px-3 rounded-lg text-xs sm:text-sm font-medium transition-all ${word === currentWord
                                                        ? 'bg-blue-600 text-white'
                                                        : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                                                        }`}
                                                >
                                                    {word}
                                                </button>
                                            ))}
                                        </div>
                                    </>
                                )}
                            </div>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>
        </>
    );
}
