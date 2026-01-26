'use client';

import { useRef, useEffect, useState, useCallback } from 'react';
import { useMediaPipe, LandmarksData } from '@/hooks/useMediaPipe';

interface CameraModuleProps {
    onLandmarks?: (landmarks: LandmarksData) => void;
    showSkeleton?: boolean;
    className?: string;
}

export function CameraModule({ onLandmarks, showSkeleton = true, className = '' }: CameraModuleProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animationFrameRef = useRef<number>();

    const [stream, setStream] = useState<MediaStream | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [videoReady, setVideoReady] = useState(false);

    const { isLoading: mediaPipeLoading, error: mediaPipeError, processFrame } = useMediaPipe({ onLandmarks });

    // Camera initialization
    useEffect(() => {
        let mounted = true;
        let currentStream: MediaStream | null = null;

        async function startCamera() {
            try {
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
                setStream(currentStream);

                const video = videoRef.current;
                if (!video) return;

                // Set up video element
                video.srcObject = currentStream;
                video.muted = true;
                video.playsInline = true;

                // Wait for video to be loaded and playing
                const playVideo = async () => {
                    try {
                        await video.play();
                        console.log('[Camera] Video playing, waiting for data...');

                        // Wait for video to have enough data
                        const checkVideoReady = () => {
                            if (video.readyState >= video.HAVE_ENOUGH_DATA) {
                                console.log('[Camera] Video ready!');
                                setVideoReady(true);
                            } else {
                                setTimeout(checkVideoReady, 100);
                            }
                        };
                        checkVideoReady();
                    } catch (err) {
                        console.error('[Camera] Play error:', err);
                        setError('Failed to start video playback. Please refresh and try again.');
                    }
                };

                // Try to play once metadata is loaded
                if (video.readyState >= video.HAVE_METADATA) {
                    playVideo();
                } else {
                    video.addEventListener('loadedmetadata', playVideo, { once: true });
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

    // Frame processing
    useEffect(() => {
        if (!videoReady || !videoRef.current || mediaPipeLoading) {
            return;
        }

        console.log('[Camera] Starting frame processing...');
        const video = videoRef.current;
        let lastProcessTime = 0;
        const FPS = 30;
        const frameInterval = 1000 / FPS;

        function processVideoFrame(timestamp: number) {
            if (!video || video.readyState !== video.HAVE_ENOUGH_DATA) {
                animationFrameRef.current = requestAnimationFrame(processVideoFrame);
                return;
            }

            if (timestamp - lastProcessTime >= frameInterval) {
                processFrame(video, timestamp);

                if (showSkeleton && canvasRef.current) {
                    const canvas = canvasRef.current;
                    const ctx = canvas.getContext('2d');
                    if (ctx && video.videoWidth > 0) {
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        // Skeleton drawing will be added here
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
    }, [videoReady, mediaPipeLoading, processFrame, showSkeleton]);

    // Error state
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

    // Loading state
    if (!videoReady || mediaPipeLoading) {
        const loadingMessage = mediaPipeLoading
            ? 'Loading MediaPipe models...'
            : !stream
                ? 'Requesting camera access...'
                : 'Starting video...';

        return (
            <div className={`flex items-center justify-center bg-gray-900 rounded-xl ${className}`}>
                <div className="text-center p-8">
                    <div className="text-6xl mb-4 animate-pulse">⏳</div>
                    <p className="text-gray-300 font-semibold text-lg">{loadingMessage}</p>
                    <p className="text-sm text-gray-400 mt-2">This may take a few seconds</p>
                    <div className="mt-4 flex justify-center">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                    </div>
                </div>
            </div>
        );
    }

    // Ready state
    return (
        <div className={`relative overflow-hidden bg-black rounded-xl ${className}`}>
            <video
                ref={videoRef}
                playsInline
                muted
                className="w-full h-full object-cover transform -scale-x-100"
            />
            {showSkeleton && (
                <canvas
                    ref={canvasRef}
                    className="absolute top-0 left-0 w-full h-full pointer-events-none transform -scale-x-100"
                />
            )}
            <div className="absolute top-4 right-4 flex items-center space-x-2 bg-black/50 backdrop-blur px-3 py-1 rounded-lg">
                <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                <span className="text-sm text-white">Live</span>
            </div>
            <div className="absolute bottom-4 left-4 bg-black/50 backdrop-blur px-3 py-1 rounded-lg text-xs text-white">
                Processing at 30 FPS
            </div>
        </div>
    );
}
