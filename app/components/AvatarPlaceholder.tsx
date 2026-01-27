'use client';

import { useState, useEffect } from 'react';

interface AvatarPlaceholderProps {
    emotion?: 'neutral' | 'happy' | 'sad' | 'angry' | 'surprised' | 'confused';
    isSpeaking?: boolean;
}

export default function AvatarPlaceholder({
    emotion = 'neutral',
    isSpeaking = false
}: AvatarPlaceholderProps) {
    const [pulseIntensity, setPulseIntensity] = useState(1);

    // Simulate speaking animation
    useEffect(() => {
        if (isSpeaking) {
            const interval = setInterval(() => {
                setPulseIntensity(0.8 + Math.random() * 0.4);
            }, 100);
            return () => clearInterval(interval);
        } else {
            setPulseIntensity(1);
        }
    }, [isSpeaking]);

    // Emotion colors
    const emotionColors: Record<string, { bg: string; accent: string; emoji: string }> = {
        neutral: { bg: 'from-blue-500 to-indigo-600', accent: 'bg-blue-400', emoji: '😐' },
        happy: { bg: 'from-green-400 to-emerald-600', accent: 'bg-green-300', emoji: '😊' },
        sad: { bg: 'from-blue-600 to-slate-700', accent: 'bg-blue-500', emoji: '😢' },
        angry: { bg: 'from-red-500 to-rose-700', accent: 'bg-red-400', emoji: '😠' },
        surprised: { bg: 'from-yellow-400 to-orange-500', accent: 'bg-yellow-300', emoji: '😮' },
        confused: { bg: 'from-purple-500 to-violet-700', accent: 'bg-purple-400', emoji: '🤔' }
    };

    const { bg, accent, emoji } = emotionColors[emotion];

    return (
        <div className="w-full h-[400px] relative overflow-hidden rounded-2xl bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700">
            {/* Background glow */}
            <div
                className={`absolute inset-0 bg-gradient-to-br ${bg} opacity-20 transition-opacity duration-500`}
                style={{ opacity: 0.2 * pulseIntensity }}
            />

            {/* Animated circles (abstract representation) */}
            <div className="absolute inset-0 flex items-center justify-center">
                {/* Outer ring */}
                <div
                    className={`absolute w-64 h-64 rounded-full bg-gradient-to-br ${bg} opacity-30 animate-pulse`}
                    style={{ transform: `scale(${pulseIntensity})` }}
                />

                {/* Middle ring */}
                <div
                    className={`absolute w-48 h-48 rounded-full bg-gradient-to-br ${bg} opacity-50`}
                    style={{
                        transform: `scale(${pulseIntensity})`,
                        animation: 'pulse 2s ease-in-out infinite'
                    }}
                />

                {/* Inner circle (head) */}
                <div
                    className={`relative w-32 h-32 rounded-full bg-gradient-to-br ${bg} shadow-2xl flex items-center justify-center`}
                    style={{ transform: `scale(${pulseIntensity})` }}
                >
                    {/* Emoji face */}
                    <span className="text-6xl">{emoji}</span>
                </div>
            </div>

            {/* Hand indicators */}
            <div className="absolute bottom-20 left-1/2 -translate-x-1/2 flex gap-8">
                {/* Left hand */}
                <div className={`w-12 h-16 rounded-lg ${accent} opacity-60 animate-bounce`}
                    style={{ animationDelay: '0ms' }} />
                {/* Right hand */}
                <div className={`w-12 h-16 rounded-lg ${accent} opacity-60 animate-bounce`}
                    style={{ animationDelay: '150ms' }} />
            </div>

            {/* Status indicator */}
            <div className="absolute top-4 left-4 flex items-center gap-2 bg-black/40 backdrop-blur px-3 py-1.5 rounded-full">
                <div className={`w-2 h-2 rounded-full ${isSpeaking ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`} />
                <span className="text-xs text-white/80">
                    {isSpeaking ? 'Signing...' : 'Ready'}
                </span>
            </div>

            {/* Emotion label */}
            <div className="absolute bottom-4 right-4 bg-black/40 backdrop-blur px-3 py-1.5 rounded-full">
                <span className="text-xs text-white/80 capitalize">{emotion}</span>
            </div>

            {/* Placeholder text */}
            <div className="absolute bottom-4 left-4 text-xs text-white/50">
                Avatar Preview (3D coming soon)
            </div>
        </div>
    );
}
