'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface TranslationPanelProps {
    currentGloss?: string;
    glossSequence?: string[];
    emotion?: string;
    confidence?: number;
    isQuestion?: boolean;
    mode: 'sign-to-text' | 'text-to-sign';
}

export default function TranslationPanel({
    currentGloss = '',
    glossSequence = [],
    emotion = 'neutral',
    confidence = 0,
    isQuestion = false,
    mode
}: TranslationPanelProps) {
    const [displayedGlosses, setDisplayedGlosses] = useState<string[]>([]);

    // Add new glosses with animation
    useEffect(() => {
        if (currentGloss && currentGloss !== displayedGlosses[displayedGlosses.length - 1]) {
            setDisplayedGlosses(prev => [...prev.slice(-5), currentGloss]); // Keep last 6
        }
    }, [currentGloss]);

    // Emotion color mapping
    const emotionColors: Record<string, string> = {
        neutral: 'bg-gray-500',
        happy: 'bg-green-500',
        sad: 'bg-blue-500',
        angry: 'bg-red-500',
        surprised: 'bg-yellow-500',
        confused: 'bg-purple-500'
    };

    return (
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 shadow-xl">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white">
                    {mode === 'sign-to-text' ? '📹 Sign → Text' : '🔤 Text → Sign'}
                </h3>

                {/* Emotion indicator */}
                <div className="flex items-center gap-2">
                    <span className={`w-3 h-3 rounded-full ${emotionColors[emotion]}`} />
                    <span className="text-sm text-white/70 capitalize">{emotion}</span>
                </div>
            </div>

            {/* Live transcription area */}
            <div className="bg-black/30 rounded-xl p-4 min-h-[120px] mb-4">
                <AnimatePresence mode="popLayout">
                    {displayedGlosses.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                            {displayedGlosses.map((gloss, index) => (
                                <motion.span
                                    key={`${gloss}-${index}`}
                                    initial={{ opacity: 0, scale: 0.8, y: 10 }}
                                    animate={{ opacity: 1, scale: 1, y: 0 }}
                                    exit={{ opacity: 0, scale: 0.8 }}
                                    transition={{ duration: 0.2 }}
                                    className={`
                    px-3 py-1.5 rounded-lg font-mono text-sm
                    ${index === displayedGlosses.length - 1
                                            ? 'bg-blue-500 text-white'
                                            : 'bg-white/20 text-white/80'}
                  `}
                                >
                                    {gloss}
                                </motion.span>
                            ))}
                        </div>
                    ) : (
                        <p className="text-white/50 text-center">
                            {mode === 'sign-to-text'
                                ? 'Start signing to see translation...'
                                : 'Type or speak to see gloss...'}
                        </p>
                    )}
                </AnimatePresence>

                {/* Question indicator */}
                {isQuestion && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="mt-2 text-xs text-yellow-400 flex items-center gap-1"
                    >
                        <span>❓</span> Question detected (non-manual markers required)
                    </motion.div>
                )}
            </div>

            {/* Confidence bar */}
            <div className="space-y-1">
                <div className="flex justify-between text-xs text-white/60">
                    <span>Confidence</span>
                    <span>{Math.round(confidence * 100)}%</span>
                </div>
                <div className="h-1.5 bg-white/20 rounded-full overflow-hidden">
                    <motion.div
                        className="h-full bg-gradient-to-r from-blue-500 to-green-500 rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${confidence * 100}%` }}
                        transition={{ duration: 0.3 }}
                    />
                </div>
            </div>

            {/* Gloss sequence preview */}
            {glossSequence.length > 0 && (
                <div className="mt-4 pt-4 border-t border-white/10">
                    <p className="text-xs text-white/50 mb-2">Gloss Sequence:</p>
                    <p className="text-sm text-white/80 font-mono">
                        {glossSequence.join(' → ')}
                    </p>
                </div>
            )}
        </div>
    );
}
