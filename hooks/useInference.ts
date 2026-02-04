'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import * as ort from 'onnxruntime-web';
import type { LandmarksData } from './useMediaPipe';

interface InferenceHook {
    predict: (landmarks: LandmarksData) => Promise<string | null>;
    isLoading: boolean;
    error: string | null;
    resetBuffer: () => void;
}

const SEQUENCE_LENGTH = 32; // Fixed sequence length for LSTM
const VOCABULARY = ['go', 'he', 'home', 'i', 'like', 'you']; // 6 words trained

export function useInference(): InferenceHook {
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [session, setSession] = useState<ort.InferenceSession | null>(null);
    const frameBuffer = useRef<number[][]>([]);
    const inputSizeRef = useRef<number>(399); // Will be determined by first frame

    // Load ONNX model
    useEffect(() => {
        const loadModel = async () => {
            try {
                setIsLoading(true);
                setError(null);

                // Configure ONNX Runtime for browser
                ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

                const modelSession = await ort.InferenceSession.create('/models/sign_model.onnx', {
                    executionProviders: ['wasm'],
                    graphOptimizationLevel: 'all'
                });

                setSession(modelSession);
                setIsLoading(false);
                console.log('✓ ONNX model loaded successfully');
            } catch (err) {
                console.error('Failed to load ONNX model:', err);
                setError(`Failed to load model: ${err instanceof Error ? err.message : 'Unknown error'}`);
                setIsLoading(false);
            }
        };

        loadModel();
    }, []);

    // Extract and flatten landmarks to match training format (399 features)
    const flattenLandmarks = useCallback((landmarks: LandmarksData): number[] => {
        const features: number[] = [];

        // Pose landmarks (33 points × 4 values = 132 features)
        // Uses pose array with visibility
        landmarks.pose.forEach(lm => {
            features.push(...lm);
        });

        // Left hand landmarks (21 points × 3 values = 63 features)
        landmarks.leftHand.forEach(lm => {
            features.push(...lm);
        });

        // Right hand landmarks (21 points × 3 values = 63 features)
        landmarks.rightHand.forEach(lm => {
            features.push(...lm);
        });

        // Face landmarks (47 points × 3 values = 141 features)
        // Training used first 47 face landmarks from essential indices
        landmarks.face.slice(0, 47).forEach(lm => {
            features.push(...lm);
        });

        return features;
    }, []);

    // Add frame and predict when buffer reaches SEQUENCE_LENGTH
    const predict = useCallback(async (landmarks: LandmarksData): Promise<string | null> => {
        if (!session) {
            return null;
        }

        try {
            // Flatten current frame
            const flatFrame = flattenLandmarks(landmarks);

            // Store feature size from first frame
            if (frameBuffer.current.length === 0) {
                inputSizeRef.current = flatFrame.length;
                console.log(`Feature size: ${flatFrame.length}`);
            }

            frameBuffer.current.push(flatFrame);

            // Wait until we have exactly SEQUENCE_LENGTH frames
            if (frameBuffer.current.length < SEQUENCE_LENGTH) {
                return null; // Not enough frames yet
            }

            // Keep only last SEQUENCE_LENGTH frames (sliding window)
            while (frameBuffer.current.length > SEQUENCE_LENGTH) {
                frameBuffer.current.shift();
            }

            // Prepare input tensor: [1, SEQUENCE_LENGTH, feature_size]
            const featureSize = inputSizeRef.current;
            const inputData = new Float32Array(frameBuffer.current.flat());
            const inputTensor = new ort.Tensor('float32', inputData, [1, SEQUENCE_LENGTH, featureSize]);

            // Run inference
            const feeds = { input: inputTensor };
            const results = await session.run(feeds);
            const output = results.output;

            // Get prediction (argmax)
            const outputData = output.data as Float32Array;
            let maxIndex = 0;
            let maxValue = outputData[0];

            for (let i = 1; i < outputData.length; i++) {
                if (outputData[i] > maxValue) {
                    maxValue = outputData[i];
                    maxIndex = i;
                }
            }

            // Apply softmax to get proper confidence
            const expValues = Array.from(outputData).map(v => Math.exp(v));
            const sumExp = expValues.reduce((a, b) => a + b, 0);
            const softmaxConfidence = expValues[maxIndex] / sumExp;

            const predictedWord = VOCABULARY[maxIndex];

            console.log(`Predicted: ${predictedWord} (confidence: ${(softmaxConfidence * 100).toFixed(1)}%)`);

            // Only return if confidence is high enough
            if (softmaxConfidence > 0.6) {
                return predictedWord;
            }

            return null;
        } catch (err) {
            console.error('Inference error:', err);
            return null;
        }
    }, [session, flattenLandmarks]);

    // Reset buffer
    const resetBuffer = useCallback(() => {
        frameBuffer.current = [];
    }, []);

    return {
        predict,
        isLoading,
        error,
        resetBuffer
    };
}
