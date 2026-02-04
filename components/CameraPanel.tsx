'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { useInference } from '@/hooks/useInference';
import { useMediaPipe } from '@/hooks/useMediaPipe';

export default function CameraPanel() {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isCameraActive, setIsCameraActive] = useState(false);
    const [isDetecting, setIsDetecting] = useState(false);
    const [sentence, setSentence] = useState<string[]>([]);
    const [lastPrediction, setLastPrediction] = useState<string | null>(null);
    const lastWordTimeRef = useRef<number>(0);
    const isDetectingRef = useRef(false);

    const { predict, isLoading: modelLoading, resetBuffer, error: modelError } = useInference();

    // Update ref when state changes
    useEffect(() => {
        isDetectingRef.current = isDetecting;
    }, [isDetecting]);

    const handleLandmarks = useCallback(async (data: any) => {
        if (!isDetectingRef.current) return;

        // Run inference
        const prediction = await predict(data);

        if (prediction) {
            // Debounce: only add word if 2 seconds passed since last word
            const now = Date.now();
            if (now - lastWordTimeRef.current > 2000) {
                setSentence(prev => [...prev, prediction]);
                setLastPrediction(prediction);
                lastWordTimeRef.current = now;
            }
        }
    }, [predict]);

    const { processFrame, isLoading: mediaPipeLoading, error: mediaPipeError } = useMediaPipe({
        onLandmarks: handleLandmarks
    });

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
                    };
                }
            } catch (err) {
                console.error('Camera error:', err);
            }
        };

        startCamera();

        return () => {
            if (videoRef.current?.srcObject) {
                const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
                tracks.forEach(track => track.stop());
            }
        };
    }, []);

    // Process video frames
    useEffect(() => {
        if (!isCameraActive || !videoRef.current || !canvasRef.current) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        let animationId: number;
        const processVideoFrame = async () => {
            if (video.readyState === video.HAVE_ENOUGH_DATA) {
                // Pass timestamp as required by useMediaPipe
                await processFrame(video, performance.now());
            }
            animationId = requestAnimationFrame(processVideoFrame);
        };

        processVideoFrame();

        return () => {
            if (animationId) cancelAnimationFrame(animationId);
        };
    }, [isCameraActive, processFrame]);

    const handleStart = () => {
        setIsDetecting(true);
        resetBuffer();
        setSentence([]);
        setLastPrediction(null);
    };

    const handleStop = () => {
        setIsDetecting(false);

        // Speak the sentence
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
                <h2 className="text-xl font-bold text-slate-800">Sign Input</h2>
                <p className="text-sm text-slate-500 mt-1">Perform signs to build your sentence</p>
            </div>

            {/* Video & Sentence Display */}
            <div className="flex-1 p-6 flex flex-col gap-4 overflow-auto">
                {/* Camera Feed */}
                <div className="relative bg-slate-900 rounded-xl overflow-hidden" style={{ aspectRatio: '4/3' }}>
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

                    {/* Status Indicator */}
                    <div className="absolute top-4 right-4 px-3 py-2 rounded-lg bg-black/60 backdrop-blur-sm">
                        {isLoading ? (
                            <span className="text-yellow-400 text-sm font-medium">⏳ Loading...</span>
                        ) : error ? (
                            <span className="text-red-400 text-sm font-medium">⚠️ Error</span>
                        ) : isDetecting ? (
                            <span className="text-green-400 text-sm font-medium flex items-center gap-2">
                                <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                                Detecting
                            </span>
                        ) : (
                            <span className="text-slate-300 text-sm font-medium">● Standby</span>
                        )}
                    </div>

                    {/* Last Prediction */}
                    {lastPrediction && isDetecting && (
                        <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 px-6 py-3 bg-blue-600 text-white rounded-xl font-bold text-lg shadow-lg animate-pulse">
                            "{lastPrediction}"
                        </div>
                    )}
                </div>

                {/* Error Display */}
                {error && (
                    <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700 text-sm">
                        {error}
                    </div>
                )}

                {/* Sentence Display */}
                <div className="bg-slate-50 rounded-xl p-4 border border-slate-200 min-h-[80px]">
                    <p className="text-xs text-slate-500 font-medium mb-2">SENTENCE:</p>
                    <p className="text-lg font-medium text-slate-800">
                        {sentence.length > 0 ? sentence.join(' ') : 'Start signing to build your sentence...'}
                    </p>
                </div>
            </div>

            {/* Controls */}
            <div className="p-6 border-t border-slate-200 bg-slate-50">
                <div className="grid grid-cols-2 gap-3">
                    {!isDetecting ? (
                        <button
                            onClick={handleStart}
                            disabled={isLoading}
                            className="col-span-2 py-4 bg-blue-600 text-white rounded-xl font-bold text-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-blue-200"
                        >
                            {isLoading ? '⏳ Loading Model...' : '▶ START DETECTION'}
                        </button>
                    ) : (
                        <>
                            <button
                                onClick={handleStop}
                                className="py-4 bg-red-500 text-white rounded-xl font-bold hover:bg-red-600 transition-all"
                            >
                                ■ STOP & SPEAK
                            </button>
                            <button
                                onClick={handleClear}
                                className="py-4 bg-slate-200 text-slate-600 rounded-xl font-medium hover:bg-slate-300 transition-all"
                            >
                                🗑️ CLEAR
                            </button>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}
