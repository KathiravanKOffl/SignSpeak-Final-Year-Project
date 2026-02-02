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
            // Model expects: 136 landmarks × 3 coords = 408 values
            // Format: upper body pose (25) + left hand (21) + right hand (21) + face subset (69) = 136 total

            // Extract ONLY x,y,z from pose (remove visibility/4th dimension)
            const poseFlat = landmarks.pose.flatMap(lm => [lm[0], lm[1], lm[2]]);  // 33 × 3 = 99
            const leftHandFlat = landmarks.leftHand.flatMap(lm => lm);  // 21 × 3 = 63
            const rightHandFlat = landmarks.rightHand.flatMap(lm => lm);  // 21 × 3 = 63
            const faceFlat = landmarks.face.flatMap(lm => lm);  // 50 × 3 = 150

            // Combine: 99 + 63 + 63 + 150 = 375 values
            // But model expects 408, so we need to match training format
            // Training used: 136 (pose) + 21 (left hand) + 21 (right hand) + 61 (face) = 136 landmarks

            // For now, send what we have and pad with zeros if needed
            const allLandmarks = [...poseFlat, ...leftHandFlat, ...rightHandFlat, ...faceFlat];

            // Pad to 408 if needed
            while (allLandmarks.length < 408) {
                allLandmarks.push(0);
            }

            // Trim to exactly 408
            const flatLandmarks = allLandmarks.slice(0, 408);

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
