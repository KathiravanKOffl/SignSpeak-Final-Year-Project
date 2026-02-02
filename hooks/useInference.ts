'use client';

import { useState, useRef } from 'react';
import type { LandmarksData } from './useMediaPipe';

interface PredictionResult {
    gloss: string;
    confidence: number;
    predictions: Array<{ class: string; confidence: number }>;
    language: string;
    processing_time_ms: number;
}

const BUFFER_SIZE = 16; // Collect 16 frames before prediction

export function useInference() {
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const frameBuffer = useRef<number[][]>([]);

    // Get API URL from environment or default to localhost
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

    // Convert single LandmarksData to flat 408-value array
    const flattenLandmarks = (landmarks: LandmarksData): number[] => {
        // Extract ONLY x,y,z from pose (remove visibility/4th dimension)
        const poseFlat = landmarks.pose.flatMap(lm => [lm[0], lm[1], lm[2]]);  // 33 × 3 = 99
        const leftHandFlat = landmarks.leftHand.flatMap(lm => lm);  // 21 × 3 = 63
        const rightHandFlat = landmarks.rightHand.flatMap(lm => lm);  // 21 × 3 = 63
        const faceFlat = landmarks.face.flatMap(lm => lm);  // 50 × 3 = 150

        const allLandmarks = [...poseFlat, ...leftHandFlat, ...rightHandFlat, ...faceFlat];

        // Pad to 408 if needed
        while (allLandmarks.length < 408) {
            allLandmarks.push(0);
        }

        // Trim to exactly 408
        return allLandmarks.slice(0, 408);
    };

    // Add frame to buffer and predict when buffer is full
    const addFrame = (landmarks: LandmarksData): boolean => {
        const flatFrame = flattenLandmarks(landmarks);
        frameBuffer.current.push(flatFrame);

        // Keep only last BUFFER_SIZE frames (sliding window)
        if (frameBuffer.current.length > BUFFER_SIZE) {
            frameBuffer.current.shift();
        }

        return frameBuffer.current.length >= BUFFER_SIZE;
    };

    const predict = async (
        landmarks: LandmarksData,
        language: 'isl' | 'asl' = 'isl'
    ): Promise<PredictionResult | null> => {
        // Add frame to buffer
        const bufferReady = addFrame(landmarks);

        // Only predict when we have enough frames
        if (!bufferReady || isLoading) {
            return null;
        }

        setIsLoading(true);
        setError(null);

        try {
            // Send all frames in buffer as a sequence
            const sequence = [...frameBuffer.current];  // Copy buffer

            console.log('[useInference] Sending sequence:', sequence.length, 'frames ×', sequence[0]?.length, 'values');

            const response = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    landmarks: sequence,  // Send full sequence (16 frames × 408 values)
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

            // Clear half the buffer for new input (sliding window)
            frameBuffer.current = frameBuffer.current.slice(BUFFER_SIZE / 2);

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

    const clearBuffer = () => {
        frameBuffer.current = [];
    };

    const getBufferSize = () => frameBuffer.current.length;

    return { predict, isLoading, error, clearBuffer, getBufferSize, BUFFER_SIZE };
}
