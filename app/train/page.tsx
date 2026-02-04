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

    const [isCameraActive, setIsCameraActive] = useState(false);
    const [status, setStatus] = useState<string>('Initializing...');
    const [showWordSelector, setShowWordSelector] = useState(true);
    const [selectedTier, setSelectedTier] = useState<string | null>(null);
    const [selectedIndividualWords, setSelectedIndividualWords] = useState<Set<string>>(new Set());

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
        setSelectedWords,
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

    // Auto-mode: immediately start next countdown after recording
    useEffect(() => {
        if (!isCountdown && !isRecording && samples.length < samplesPerWord && !isPaused && !showWordSelector) {
            const timer = setTimeout(() => {
                startCountdown();
            }, 500);
            return () => clearTimeout(timer);
        }
    }, [isCountdown, isRecording, samples.length, samplesPerWord, isPaused, showWordSelector]);

    const handleStart = () => {
        frameBufferRef.current = [];
        recordingStartRef.current = Date.now();
        startRecording();
        setStatus('Recording...');
    };

    const handleAutoStop = () => {
        stopRecording();
        const normalized = resampleToN(frameBufferRef.current, 32);

        // Auto-confirm in auto mode (no preview needed)
        confirmSample(normalized);
        setStatus(`Saved! (${samples.length + 1}/${samplesPerWord})`);
    };

    const handleStartTraining = () => {
        const words = selectedTier
            ? TIERS[selectedTier as keyof typeof TIERS]
            : Array.from(selectedIndividualWords);

        if (words.length === 0) {
            alert('Please select at least one word or tier');
            return;
        }

        setSelectedWords(words);
        setShowWordSelector(false);
        setTimeout(() => startCountdown(), 500);
    };

    const toggleIndividualWord = (word: string) => {
        const newSet = new Set(selectedIndividualWords);
        if (newSet.has(word)) {
            newSet.delete(word);
        } else {
            newSet.add(word);
        }
        setSelectedIndividualWords(newSet);
        setSelectedTier(null); // Clear tier if selecting individual words
    };

    const handleTierSelect = (tier: string) => {
        setSelectedTier(tier);
        setSelectedIndividualWords(new Set()); // Clear individual words
    };

    // Word Selection Modal
    if (showWordSelector) {
        return (
            <main className="min-h-screen bg-[#F8F9FA] flex items-center justify-center p-4">
                <div className="bg-white rounded-2xl shadow-lg max-w-2xl w-full p-6 max-h-[90vh] overflow-y-auto">
                    <h1 className="text-3xl font-bold text-slate-800 mb-2">Select Words to Train</h1>
                    <p className="text-slate-500 text-sm mb-6">Choose a tier or select individual words</p>

                    {/* Tier Selection */}
                    <div className="mb-6">
                        <h2 className="text-lg font-bold text-slate-700 mb-3">Quick Select by Tier</h2>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                            {Object.entries(TIERS).map(([name, words]) => (
                                <button
                                    key={name}
                                    onClick={() => handleTierSelect(name)}
                                    className={`p-4 rounded-xl border-2 text-left transition-all ${selectedTier === name
                                            ? 'border-blue-600 bg-blue-50'
                                            : 'border-slate-200 hover:border-blue-300'
                                        }`}
                                >
                                    <div className="font-bold text-slate-800">{name.split(':')[0]}</div>
                                    <div className="text-sm text-slate-500 mt-1">{words.length} words</div>
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="border-t pt-6">
                        <h2 className="text-lg font-bold text-slate-700 mb-3">Or Select Individual Words</h2>
                        <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 gap-2">
                            {FULL_VOCABULARY.map(word => (
                                <button
                                    key={word}
                                    onClick={() => toggleIndividualWord(word)}
                                    className={`py-2 px-3 rounded-lg text-sm font-medium transition-all ${selectedIndividualWords.has(word)
                                            ? 'bg-blue-600 text-white'
                                            : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                                        }`}
                                >
                                    {word}
                                </button>
                            ))}
                        </div>
                    </div>

                    <button
                        onClick={handleStartTraining}
                        className="w-full mt-6 py-4 bg-blue-600 text-white rounded-xl font-bold text-lg hover:bg-blue-700 shadow-lg"
                    >
                        Start Training ({selectedTier ? TIERS[selectedTier as keyof typeof TIERS].length : selectedIndividualWords.size} words)
                    </button>
                </div>
            </main>
        );
    }

    return (
        <main className="min-h-screen bg-[#F8F9FA] flex flex-col">
            <div className="p-4 text-center border-b bg-white">
                <h1 className="text-2xl font-bold text-slate-800">Training - Auto Mode</h1>
                <p className="text-slate-500 text-xs">Recording {samplesPerWord} samples per word automatically</p>
            </div>

            <div className="flex-1 flex flex-col lg:grid lg:grid-cols-2 gap-4 p-4 max-w-7xl mx-auto w-full">

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
                                className="text-9xl font-bold text-white"
                            >
                                {countdownValue}
                            </motion.div>
                        </div>
                    )}

                    {/* Recording indicator */}
                    {isRecording && (
                        <div className="absolute top-4 right-4 bg-red-500 text-white px-3 py-2 rounded-full text-sm font-bold flex items-center gap-2 z-10">
                            <div className="w-3 h-3 bg-white rounded-full animate-pulse" />
                            REC ({Math.ceil((100 - recordingProgress) / 50)}s)
                        </div>
                    )}

                    {isRecording && (
                        <div className="absolute bottom-0 left-0 right-0 h-2 bg-slate-200 z-10">
                            <motion.div className="h-full bg-red-500" animate={{ width: `${recordingProgress}%` }} />
                        </div>
                    )}

                    {(isModelLoading || !isCameraActive) && (
                        <div className="absolute inset-0 bg-white/90 flex items-center justify-center z-20">
                            <div className="text-center">
                                <div className="w-12 h-12 border-4 border-slate-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-4" />
                                <p className="text-slate-600 font-medium">{status}</p>
                            </div>
                        </div>
                    )}
                </div>

                {/* Controls */}
                <div className="flex flex-col gap-4">
                    <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl shadow-lg p-6 text-white">
                        <p className="text-xs uppercase tracking-wider mb-1 opacity-90">Current Word</p>
                        <h2 className="text-5xl font-bold mb-4">{currentWord}</h2>
                        <div className="flex justify-between text-sm">
                            <span>{samples.length}/{samplesPerWord} samples</span>
                            <span>Word {currentWordIndex + 1}/{vocabulary.length}</span>
                        </div>
                    </div>

                    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-4">
                        <div className="flex justify-between text-xs text-slate-600 mb-2">
                            <span>Progress</span>
                            <span>{Math.round((samples.length / samplesPerWord) * 100)}%</span>
                        </div>
                        <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                            <motion.div className="h-full bg-green-500" animate={{ width: `${(samples.length / samplesPerWord) * 100}%` }} />
                        </div>
                    </div>

                    {/* Pause/Resume/Retry Controls */}
                    <div className="grid grid-cols-2 gap-3">
                        {!isRecording && !isCountdown && (
                            <button onClick={startCountdown} disabled={!isCameraActive || isModelLoading} className="col-span-2 py-5 bg-blue-600 text-white rounded-xl font-bold text-xl hover:bg-blue-700 disabled:opacity-50 shadow-lg">
                                ▶ START
                            </button>
                        )}

                        {isCountdown && (
                            <>
                                <button onClick={isPaused ? resumeTraining : pauseTraining} className="py-4 bg-yellow-500 text-white rounded-xl font-bold">
                                    {isPaused ? '▶ RESUME' : '⏸ PAUSE'}
                                </button>
                                {isPaused && (
                                    <button onClick={() => { retrySample(); setIsCountdown(false); if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current); }} className="py-4 bg-red-500 text-white rounded-xl font-bold">
                                        ↺ RETRY
                                    </button>
                                )}
                                {!isPaused && (
                                    <button onClick={() => { setIsCountdown(false); if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current); }} className="py-4 bg-red-500 text-white rounded-xl font-bold">
                                        ✕ CANCEL
                                    </button>
                                )}
                            </>
                        )}
                    </div>

                    {/* Reset Word */}
                    <button onClick={resetWord} className="py-3 bg-white border-2 border-slate-200 text-slate-600 rounded-xl font-medium hover:bg-slate-50">
                        Reset Word
                    </button>

                    {/* Status */}
                    <div className="bg-white rounded-xl border border-slate-200 p-3 text-center">
                        <div className="flex items-center justify-center gap-2">
                            <div className={`w-2 h-2 rounded-full ${isCameraActive ? 'bg-green-500 animate-pulse' : 'bg-slate-300'}`} />
                            <span className="text-xs font-medium text-slate-700">{status}</span>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    );
}
