'use client';

import { useRef, useEffect, useState } from 'react';
import { useMediaPipe, LandmarksData } from '@/hooks/useMediaPipe';

interface CameraModuleProps {
    onLandmarks?: (landmarks: LandmarksData) => void;
    showSkeleton?: boolean;
    className?: string;
}

export function CameraModule({ onLandmarks, showSkeleton = true, className = '' }: CameraModuleProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [stream, setStream] = useState<MediaStream | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [cameraReady, setCameraReady] = useState(false);
    const animationFrameRef = useRef<number>();

    const { isLoading, error: mediaPipeError, processFrame } = useMediaPipe({ onLandmarks });

    // Initialize camera
    useEffect(() => {
        let mounted = true;
        let mediaStream: MediaStream | null = null;

        async function initCamera() {
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                        facingMode: 'user',
                    },
                    audio: false,
                });

                if (mounted && videoRef.current) {
                    videoRef.current.srcObject = mediaStream;
                    setStream(mediaStream);

                    // Wait for video to be ready and play it
                    videoRef.current.onloadedmetadata = async () => {
                        if (videoRef.current && mounted) {
                            try {
                                await videoRef.current.play();
                                setCameraReady(true);
                            } catch (playError) {
                                console.error('Video play error:', playError);
                                setError('Failed to start video playback');
                            }
                        }
                    };
                }
            } catch (err) {
                console.error('Camera access error:', err);
                if (mounted) {
                    setError(err instanceof Error ? err.message : 'Failed to access camera');
                }
            }
        }

        initCamera();

        return () => {
            mounted = false;
            // Stop tracks from the locally captured mediaStream
            mediaStream?.getTracks().forEach((track) => track.stop());
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
        };
    }, []); // Empty deps - only run once on mount

    // Process frames
    useEffect(() => {
        if (!videoRef.current || isLoading || !stream || !cameraReady) {
            return;
        }

        const video = videoRef.current;
        let lastProcessTime = 0;
        const FPS = 30;
        const frameInterval = 1000 / FPS;

        function processVideoFrame(timestamp: number) {
            if (!video || video.readyState !== video.HAVE_ENOUGH_DATA) {
                animationFrameRef.current = requestAnimationFrame(processVideoFrame);
                return;
            }

            // Throttle to target FPS
            if (timestamp - lastProcessTime >= frameInterval) {
                processFrame(video, timestamp);

                // Draw skeleton overlay
                if (showSkeleton && canvasRef.current) {
                    const canvas = canvasRef.current;
                    const ctx = canvas.getContext('2d');
                    if (ctx) {
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        // Drawing logic will be added when we have landmarks
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
    }, [isLoading, stream, processFrame, showSkeleton, cameraReady]);

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

    if (isLoading || !cameraReady) {
        return (
            <div className={`flex items-center justify-center bg-gray-900 rounded-xl ${className}`}>
                <div className="text-center p-8">
                    <div className="text-6xl mb-4 animate-pulse">⏳</div>
                    <p className="text-gray-300 font-semibold text-lg">
                        {isLoading ? 'Loading MediaPipe models...' : 'Initializing camera...'}
                    </p>
                    <p className="text-sm text-gray-400 mt-2">This may take a few seconds</p>
                    <div className="mt-4 flex justify-center">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className={`relative overflow-hidden bg-black rounded-xl ${className}`}>
            <video
                ref={videoRef}
                autoPlay
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
