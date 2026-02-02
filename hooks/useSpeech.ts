'use client';

import { useEffect, useState } from 'react';

export function useSpeech() {
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
    const [selectedVoice, setSelectedVoice] = useState<SpeechSynthesisVoice | null>(null);
    const [isSupported, setIsSupported] = useState(false);

    useEffect(() => {
        // Check if speech synthesis is supported
        if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
            setIsSupported(true);

            // Load voices
            const loadVoices = () => {
                const availableVoices = window.speechSynthesis.getVoices();
                setVoices(availableVoices);

                // Default to first English voice
                const englishVoice = availableVoices.find(v => v.lang.startsWith('en'));
                setSelectedVoice(englishVoice || availableVoices[0] || null);

                console.log('[useSpeech] Loaded', availableVoices.length, 'voices');
            };

            loadVoices();
            window.speechSynthesis.onvoiceschanged = loadVoices;
        } else {
            console.warn('[useSpeech] Speech synthesis not supported');
        }
    }, []);

    const speak = (text: string) => {
        if (!isSupported || !text) return;

        // Cancel any ongoing speech
        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);

        if (selectedVoice) {
            utterance.voice = selectedVoice;
        }

        utterance.rate = 0.9;    // Slightly slower for clarity
        utterance.pitch = 1.0;
        utterance.volume = 1.0;

        utterance.onstart = () => {
            setIsSpeaking(true);
            console.log('[useSpeech] Speaking:', text);
        };

        utterance.onend = () => {
            setIsSpeaking(false);
            console.log('[useSpeech] Finished');
        };

        utterance.onerror = (event) => {
            setIsSpeaking(false);
            console.error('[useSpeech] Error:', event);
        };

        window.speechSynthesis.speak(utterance);
    };

    const stop = () => {
        if (!isSupported) return;
        window.speechSynthesis.cancel();
        setIsSpeaking(false);
    };

    return { speak, stop, isSpeaking, voices, selectedVoice, setSelectedVoice, isSupported };
}
