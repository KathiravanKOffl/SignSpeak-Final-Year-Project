'use client';

import { useSearchParams } from 'next/navigation';
import { Suspense, useEffect, useRef, useCallback, useState } from 'react';
import { CameraModule } from '@/components/camera/CameraModule';
import { TranscriptPanel } from '@/components/transcript/TranscriptPanel';
import { useAppStore } from '@/stores/appStore';
import { useInference } from '@/hooks/useInference';
import { useSpeech } from '@/hooks/useSpeech';
import type { LandmarksData } from '@/hooks/useMediaPipe';

function AppContent() {
    const searchParams = useSearchParams();
    const language = (searchParams.get('lang') || 'isl') as 'isl' | 'asl';
    const [predictionStatus, setPredictionStatus] = useState<string>('');
    const lastPredictionTime = useRef<number>(0);
    const landmarkBuffer = useRef<LandmarksData[]>([]);

    const {
        setLanguage,
        setCurrentLandmarks,
        transcript,
        clearTranscript,
        addTranscriptMessage,
    } = useAppStore();

    // Initialize hooks
    const { predict, isLoading: predictLoading, error: predictError } = useInference();
    const { speak, isSpeaking } = useSpeech();

    // Set language on mount
    useEffect(() => {
        setLanguage(language);
    }, [language, setLanguage]);

    // Send prediction to backend (throttled)
    const sendPrediction = useCallback(async (landmarks: LandmarksData) => {
        const result = await predict(landmarks, language);

        if (result && result.gloss) {
            const conf = result.confidence || 0;
            setPredictionStatus(`${result.gloss} (${(conf * 100).toFixed(0)}%)`);

            // Add to transcript with confidence threshold
            if (conf > 0.3) {  // Only show confident predictions
                addTranscriptMessage({
                    id: Date.now().toString(),
                    text: result.gloss,
                    type: 'sign',
                    timestamp: new Date(),
                    confidence: conf,
                });

                // Speak the prediction
                speak(result.gloss);
            }
        } else if (predictError) {
            setPredictionStatus(`Error: ${predictError}`);
        }
    }, [language, predict, speak, addTranscriptMessage, predictError]);

    const handleLandmarks = useCallback((landmarks: LandmarksData) => {
        setCurrentLandmarks(landmarks);

        // Throttle predictions to every 500ms
        const now = Date.now();
        const timeSinceLast = now - lastPredictionTime.current;

        // Debug logging every 2 seconds
        if (timeSinceLast >= 2000) {
            console.log('[Landmarks] Confidence:', landmarks.confidence.toFixed(2),
                'Hands:', landmarks.leftHand.some(l => l[0] !== 0) || landmarks.rightHand.some(l => l[0] !== 0));
        }

        if (timeSinceLast >= 500 && landmarks.confidence > 0.3) {
            console.log('[Landmarks] Sending prediction...');
            lastPredictionTime.current = now;
            sendPrediction(landmarks);
        }
    }, [setCurrentLandmarks, sendPrediction]);

    return (
        <div className="min-h-screen bg-gray-900 text-white">
            {/* Header */}
            <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                        <h1 className="text-2xl font-bold">SignSpeak</h1>
                        <span className="px-3 py-1 bg-blue-600 rounded-full text-sm">
                            {language.toUpperCase()}
                        </span>
                    </div>
                    <div className="flex items-center space-x-4">
                        <button className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition">
                            Settings
                        </button>
                        <button className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg transition">
                            📱 Enable Multi-Device
                        </button>
                    </div>
                </div>
            </header>

            {/* Main Content - Split Screen */}
            <main className="h-[calc(100vh-80px)] flex flex-col">
                {/* Video & Avatar Area */}
                <div className="flex-1 grid md:grid-cols-2 gap-4 p-4">
                    {/* Camera Feed */}
                    <div className="relative">
                        <div className="absolute top-4 left-4 z-10">
                            <div className="bg-black/50 px-3 py-1 rounded-lg text-sm">
                                📷 Live Camera
                            </div>
                        </div>
                        <CameraModule
                            onLandmarks={handleLandmarks}
                            showSkeleton={false}
                            className="w-full h-full"
                        />
                    </div>

                    {/* Avatar Display */}
                    <div className="bg-gray-800 rounded-xl overflow-hidden relative">
                        <div className="absolute top-4 left-4 z-10">
                            <div className="bg-black/50 px-3 py-1 rounded-lg text-sm">
                                🤖 3D Avatar
                            </div>
                        </div>
                        <div className="w-full h-full flex items-center justify-center text-gray-500">
                            <div className="text-center">
                                <div className="text-6xl mb-4">🧑</div>
                                <p>3D Avatar will appear here</p>
                                <p className="text-sm mt-2">With sign animations</p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Transcript Panel */}
                <TranscriptPanel messages={transcript} onClear={clearTranscript} />
            </main>
        </div>
    );
}

export default function AppPage() {
    return (
        <Suspense fallback={<div>Loading...</div>}>
            <AppContent />
        </Suspense>
    );
}
