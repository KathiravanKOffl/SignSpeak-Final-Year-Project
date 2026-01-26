'use client';

import { useRef, useEffect, useState, useCallback } from 'react';
import { useMediaPipe, LandmarksData } from '@/hooks/useMediaPipe';
import { drawSkeleton } from './SkeletonDrawing';

interface CameraModuleProps {
    onLandmarks?: (landmarks: LandmarksData) => void;
    showSkeleton?: boolean;
    className?: string;
}

export function CameraModule({ onLandmarks, showSkeleton = true, className = '' }: CameraModuleProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animationFrameRef = useRef<number>();
    const landmarksRef = useRef<LandmarksData | null>(null);

    const [stream, setStream] = useState<MediaStream | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [cameraStatus, setCameraStatus] = useState<'idle' | 'requesting' | 'granted' | 'playing' | 'ready'>('idle');
    const [debugInfo, setDebugInfo] = useState({ confidence: 0, hasHands: false });

    // Handle landmarks - store locally and pass to parent
    const handleLandmarks = useCallback((landmarks: LandmarksData) => {
        landmarksRef.current = landmarks;
        setDebugInfo({
            confidence: landmarks.confidence,
            hasHands: landmarks.leftHand.some(lm => lm[0] !== 0) || landmarks.rightHand.some(lm => lm[0] !== 0)
        });
        onLandmarks?.(landmarks);
    }, [onLandmarks]);

    const { isLoading: mediaPipeLoading, error: mediaPipeError, processFrame } = useMediaPipe({
        onLandmarks: handleLandmarks
    });

    // Camera initialization
    useEffect(() => {
        if (!videoRef.current) {
            console.log('[Camera] Video element not ready, waiting...');
            return;
        }

        let mounted = true;
        let currentStream: MediaStream | null = null;
        const video = videoRef.current;

        async function startCamera() {
            try {
                setCameraStatus('requesting');
                console.log('[Camera] Requesting camera access...');

                currentStream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                        facingMode: 'user',
                    },
                    audio: false,
                });

                if (!mounted) {
                    currentStream.getTracks().forEach(track => track.stop());
                    return;
                }

                console.log('[Camera] Camera access granted');
                setCameraStatus('granted');
                setStream(currentStream);

                video.srcObject = currentStream;
                video.muted = true;
                video.playsInline = true;
                console.log('[Camera] Stream attached to video element');

                const handleCanPlay = async () => {
                    console.log('[Camera] Video can play, starting...');
                    try {
                        await video.play();
                        console.log('[Camera] Video playing!');
                        setCameraStatus('playing');

                        const checkReady = () => {
                            if (video.readyState >= video.HAVE_ENOUGH_DATA) {
                                console.log('[Camera] ✅ Video ready with enough data!');
                                setCameraStatus('ready');
                            } else {
                                console.log(`[Camera] Waiting for data... readyState=${video.readyState}`);
                                setTimeout(checkReady, 100);
                            }
                        };
                        checkReady();
                    } catch (err) {
                        console.error('[Camera] Play error:', err);
                        video.muted = true;
                        try {
                            await video.play();
                            console.log('[Camera] Video playing (muted retry)!');
                            setCameraStatus('ready');
                        } catch (err2) {
                            setError('Failed to start video. Please click the page and refresh.');
                        }
                    }
                };

                video.addEventListener('canplay', handleCanPlay, { once: true });

                if (video.readyState >= video.HAVE_FUTURE_DATA) {
                    video.removeEventListener('canplay', handleCanPlay);
                    handleCanPlay();
                }

            } catch (err) {
                console.error('[Camera] Access error:', err);
                if (mounted) {
                    setError(err instanceof Error ? err.message : 'Failed to access camera');
                }
            }
        }

        startCamera();

        return () => {
            mounted = false;
            console.log('[Camera] Cleaning up...');
            currentStream?.getTracks().forEach(track => track.stop());
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
        };
    }, []);

    // Frame processing + skeleton drawing
    useEffect(() => {
        if (cameraStatus !== 'ready' || !videoRef.current || mediaPipeLoading) {
            return;
        }

        console.log('[Camera] Starting frame processing...');
        const video = videoRef.current;
        const canvas = canvasRef.current;
        let lastProcessTime = 0;
        const FPS = 30;
        const frameInterval = 1000 / FPS;

        function processVideoFrame(timestamp: number) {
            if (!video || video.readyState !== video.HAVE_ENOUGH_DATA) {
                animationFrameRef.current = requestAnimationFrame(processVideoFrame);
                return;
            }

            if (timestamp - lastProcessTime >= frameInterval) {
                // Process landmarks
                processFrame(video, timestamp);

                // Draw skeleton on canvas
                if (showSkeleton && canvas && landmarksRef.current) {
                    const ctx = canvas.getContext('2d');
                    if (ctx && video.videoWidth > 0) {
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;

                        drawSkeleton(ctx, landmarksRef.current, canvas.width, canvas.height);
                    }
                }

                lastProcessTime = timestamp;
            }

            animationFrameRef.current = requestAnimationFrame(processVideoFrame);
        }

        animationFrameRef.current = requestAnimationFrame(processVideoFrame);

        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
        };
    }, [cameraStatus, mediaPipeLoading, processFrame, showSkeleton]);

    const getLoadingMessage = () => {
        if (mediaPipeLoading) return 'Loading AI models...';
        switch (cameraStatus) {
            case 'idle': return 'Initializing...';
            case 'requesting': return 'Requesting camera access...';
            case 'granted': return 'Setting up video...';
            case 'playing': return 'Almost ready...';
            default: return 'Loading...';
        }
    };

    const isLoading = cameraStatus !== 'ready' || mediaPipeLoading;

    if (error || mediaPipeError) {
        return (
            <div className={`flex items-center justify-center bg-gray-800 rounded-xl ${className}`}>
                <div className="text-center p-8">
                    <div className="text-6xl mb-4">⚠️</div>
                    <p className="text-red-400">{error || mediaPipeError}</p>
                    <p className="text-sm text-gray-400 mt-2">Please check camera permissions</p>
                </div>
            </div>
        );
    }

    return (
        <div className={`relative overflow-hidden bg-black rounded-xl ${className}`}>
            {/* Video element - ALWAYS in DOM */}
            <video
                ref={videoRef}
                playsInline
                muted
                className={`w-full h-full object-cover transform -scale-x-100 ${isLoading ? 'opacity-0' : 'opacity-100'}`}
                style={{ transition: 'opacity 0.3s ease' }}
            />

            {/* Canvas overlay for skeleton - mirrors the video flip */}
            {showSkeleton && !isLoading && (
                <canvas
                    ref={canvasRef}
                    className="absolute top-0 left-0 w-full h-full pointer-events-none transform -scale-x-100"
                />
            )}

            {/* Loading overlay */}
            {isLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
                    <div className="text-center p-8">
                        <div className="text-6xl mb-4 animate-pulse">⏳</div>
                        <p className="text-gray-300 font-semibold text-lg">{getLoadingMessage()}</p>
                        <p className="text-sm text-gray-400 mt-2">This may take a few seconds</p>
                        <div className="mt-4 flex justify-center">
                            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                        </div>
                    </div>
                </div>
            )}

            {/* Live indicator + debug info */}
            {!isLoading && (
                <>
                    <div className="absolute top-4 right-4 flex items-center space-x-2 bg-black/50 backdrop-blur px-3 py-1 rounded-lg">
                        <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                        <span className="text-sm text-white">Live</span>
                    </div>
                    <div className="absolute bottom-4 left-4 bg-black/50 backdrop-blur px-3 py-1 rounded-lg text-xs text-white space-y-1">
                        <div>Confidence: {(debugInfo.confidence * 100).toFixed(0)}%</div>
                        <div>Hands: {debugInfo.hasHands ? '✅ Detected' : '❌ Not detected'}</div>
                    </div>
                </>
            )}
        </div>
    );
}
