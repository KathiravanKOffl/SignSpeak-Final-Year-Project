'use client';

import { useState, useRef, useCallback, useEffect } from 'react';
import type { LandmarksData } from './useMediaPipe';

interface InferenceHook {
    predict: (landmarks: LandmarksData) => Promise<string | null>;
    isLoading: boolean;
    error: string | null;
    resetBuffer: () => void;
    vocabulary: string[];
    isConnected: boolean;
    testConnection: () => Promise<boolean>;
}

const SEQUENCE_LENGTH = 32;

export function useInference(): InferenceHook {
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [vocabulary, setVocabulary] = useState<string[]>([]);
    const [isConnected, setIsConnected] = useState(false);
    const frameBuffer = useRef<number[][]>([]);

    // Get backend URL from localStorage
    const getBackendUrl = (): string | null => {
        if (typeof window === 'undefined') return null;
        const url = localStorage.getItem('backend_url');
        // Remove trailing slash to prevent double-slash in URLs
        return url ? url.replace(/\/$/, '') : null;
    };

    // Flatten landmarks into feature array (399 features)
    const flattenLandmarks = (landmarks: LandmarksData): number[] => {
        const features: number[] = [];

        // Pose landmarks (33 points × 4 values = 132 features)
        landmarks.pose.forEach(lm => features.push(...lm));

        // Left hand landmarks (21 points × 3 values = 63 features)
        landmarks.leftHand.forEach(lm => features.push(...lm));

        // Right hand landmarks (21 points × 3 values = 63 features)
        landmarks.rightHand.forEach(lm => features.push(...lm));

        // Face landmarks (47 points × 3 values = 141 features)
        landmarks.face.slice(0, 47).forEach(lm => features.push(...lm));

        return features;
    };

    // Test connection to backend
    const testConnection = useCallback(async (): Promise<boolean> => {
        const backendUrl = getBackendUrl();
        if (!backendUrl) {
            setError('No backend URL configured');
            setIsConnected(false);
            return false;
        }

        try {
            setIsLoading(true);
            setError(null);

            const response = await fetch(`${backendUrl}/`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                throw new Error(`Backend returned ${response.status}`);
            }

            const data = await response.json();

            if (data.vocabulary && Array.isArray(data.vocabulary)) {
                setVocabulary(data.vocabulary);
                setIsConnected(true);
                console.log('✓ Connected to backend:', data);
                return true;
            } else {
                throw new Error('Invalid response from backend');
            }
        } catch (err) {
            console.error('Connection test failed:', err);
            setError(`Connection failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
            setIsConnected(false);
            return false;
        } finally {
            setIsLoading(false);
        }
    }, []);

    // Auto-connect on mount if URL exists
    useEffect(() => {
        const backendUrl = getBackendUrl();
        if (backendUrl) {
            testConnection();
        }
    }, [testConnection]);

    // Predict sign from landmarks
    const predict = useCallback(async (landmarks: LandmarksData): Promise<string | null> => {
        const backendUrl = getBackendUrl();
        if (!backendUrl) {
            setError('Backend URL not configured');
            return null;
        }

        // Add frame to buffer
        const flatFrame = flattenLandmarks(landmarks);
        frameBuffer.current.push(flatFrame);

        // Need full sequence
        if (frameBuffer.current.length < SEQUENCE_LENGTH) {
            return null;
        }

        // Keep only last 32 frames
        while (frameBuffer.current.length > SEQUENCE_LENGTH) {
            frameBuffer.current.shift();
        }

        try {
            setIsLoading(true);

            const response = await fetch(`${backendUrl}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sequence: frameBuffer.current })
            });

            if (!response.ok) {
                throw new Error(`Prediction failed: ${response.status}`);
            }

            const result = await response.json();

            // Return word if confidence is high enough
            if (result.word && result.confidence > 0.6) {
                return result.word;
            }

            return null;
        } catch (err) {
            console.error('Prediction error:', err);
            setError(`Prediction failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
            return null;
        } finally {
            setIsLoading(false);
        }
    }, []);

    // Reset frame buffer
    const resetBuffer = useCallback(() => {
        frameBuffer.current = [];
    }, []);

    return {
        predict,
        isLoading,
        error,
        resetBuffer,
        vocabulary,
        isConnected,
        testConnection
    };
}
