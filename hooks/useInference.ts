'use client';

import { useState } from 'react';
import type { LandmarksData } from './useMediaPipe';

interface PredictionResult {
    gloss: string;
    confidence: number;
    predictions: Array<{ class: string; confidence: number }>;
    language: string;
    processing_time_ms: number;
}

export function useInference() {
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Get API URL from environment or default to localhost
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

    const predict = async (
        landmarks: LandmarksData,
        language: 'isl' | 'asl' = 'isl'
    ): Promise<PredictionResult | null> => {
        setIsLoading(true);
        setError(null);

        try {
            // Convert landmarks to flat array format expected by backend
            // LandmarksData has: leftHand, rightHand, pose (each is number[][])
            const flatLandmarks = [
                ...landmarks.leftHand.flatMap(lm => lm),  // Flatten each hand
                ...landmarks.rightHand.flatMap(lm => lm),
                ...landmarks.pose.flatMap(lm => lm)
            ];

            console.log('[useInference] Sending landmarks:', flatLandmarks.length, 'values');

            const response = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    landmarks: [flatLandmarks],  // Wrap in array for batch processing
                    language,
                    top_k: 5
                })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
                throw new Error(errorData.error || `API error: ${response.status}`);
            }

            const result = await response.json();
            console.log('[useInference] Prediction:', result);

            return result;

        } catch (err) {
            const message = err instanceof Error ? err.message : 'Prediction failed';
            setError(message);
            console.error('[useInference] Error:', err);
            return null;
        } finally {
            setIsLoading(false);
        }
    };

    return { predict, isLoading, error };
}
