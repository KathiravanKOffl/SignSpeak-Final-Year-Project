'use client';

import { useState, useRef, useEffect } from 'react';
import * as ort from 'onnxruntime-web';
import type { LandmarksData } from './useMediaPipe';

interface InferenceHook {
    predict: (landmarks: LandmarksData) => Promise<string | null>;
    isLoading: boolean;
    error: string | null;
    resetBuffer: () => void;
}

const SEQUENCE_LENGTH = 32; // Fixed sequence length for LSTM
const VOCABULARY = ['he', 'i', 'you']; // Update this when you add more words

export function useInference(): InferenceHook {
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [session, setSession] = useState<ort.InferenceSession | null>(null);
    const frameBuffer = useRef<number[][]>([]);

    // Load ONNX model
    useEffect(() => {
        const loadModel = async () => {
            try {
                setIsLoading(true);
                const modelSession = await ort.InferenceSession.create('/models/sign_model.onnx');
                setSession(modelSession);
                setIsLoading(false);
                console.log('✓ ONNX model loaded successfully');
            } catch (err) {
                console.error('Failed to load ONNX model:', err);
                setError('Failed to load model');
                setIsLoading(false);
            }
        };

        loadModel();
    }, []);

    // Extract and flatten landmarks to 399 features
    const flattenLandmarks = (landmarks: LandmarksData): number[] => {
        const features: number[] = [];

        // Pose landmarks (33 points × 4 values = 132 features)
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
        // MediaPipe provides 468 face landmarks, but we only saved 47
        landmarks.face.slice(0, 47).forEach(lm => {
            features.push(...lm);
        });

        return features;
    };

    // Add frame and predict when buffer reaches SEQUENCE_LENGTH
    const predict = async (landmarks: LandmarksData): Promise<string | null> => {
        if (!session) {
            console.warn('Model not loaded yet');
            return null;
        }

        try {
            // Flatten current frame
            const flatFrame = flattenLandmarks(landmarks);
            frameBuffer.current.push(flatFrame);

            // Wait until we have exactly SEQUENCE_LENGTH frames
            if (frameBuffer.current.length < SEQUENCE_LENGTH) {
                return null; // Not enough frames yet
            }

            // Keep only last SEQUENCE_LENGTH frames (sliding window)
            if (frameBuffer.current.length > SEQUENCE_LENGTH) {
                frameBuffer.current.shift();
            }

            // Prepare input tensor: [1, SEQUENCE_LENGTH, 399]
            const inputData = new Float32Array(frameBuffer.current.flat());
            const inputTensor = new ort.Tensor('float32', inputData, [1, SEQUENCE_LENGTH, 399]);

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

            const predictedWord = VOCABULARY[maxIndex];
            const confidence = maxValue;

            console.log(`Predicted: ${predictedWord} (confidence: ${(confidence * 100).toFixed(1)}%)`);

            // Only return if confidence is high enough
            if (confidence > 0.7) {
                return predictedWord;
            }

            return null;
        } catch (err) {
            console.error('Inference error:', err);
            setError('Prediction failed');
            return null;
        }
    };

    // Reset buffer
    const resetBuffer = () => {
        frameBuffer.current = [];
    };

    return {
        predict,
        isLoading,
        error,
        resetBuffer
    };
}
