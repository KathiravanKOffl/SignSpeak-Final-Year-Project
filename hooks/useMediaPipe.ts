'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { PoseLandmarker, HandLandmarker, FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

export interface LandmarksData {
    pose: number[][];
    leftHand: number[][];
    rightHand: number[][];
    face: number[][];
    // Raw coordinates (0-1 range) for drawing on screen
    rawLeftHand: number[][];
    rawRightHand: number[][];
    confidence: number;
    timestamp: number;
}

interface UseMediaPipeOptions {
    onLandmarks?: (landmarks: LandmarksData) => void;
    minDetectionConfidence?: number;
    minTrackingConfidence?: number;
}

export function useMediaPipe(options: UseMediaPipeOptions = {}) {
    const {
        onLandmarks,
        minDetectionConfidence = 0.5,
        minTrackingConfidence = 0.5,
    } = options;

    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const isProcessingRef = useRef(false);

    const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
    const handLandmarkerRef = useRef<HandLandmarker | null>(null);
    const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);

    // Initialize MediaPipe models
    useEffect(() => {
        let mounted = true;

        async function init() {
            try {
                const vision = await FilesetResolver.forVisionTasks(
                    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
                );

                // Initialize Pose Landmarker
                const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
                    baseOptions: {
                        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
                        delegate: 'GPU',
                    },
                    runningMode: 'VIDEO',
                    numPoses: 1,
                    minPoseDetectionConfidence: minDetectionConfidence,
                    minTrackingConfidence: minTrackingConfidence,
                });

                // Initialize Hand Landmarker
                const handLandmarker = await HandLandmarker.createFromOptions(vision, {
                    baseOptions: {
                        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
                        delegate: 'GPU',
                    },
                    runningMode: 'VIDEO',
                    numHands: 2,
                    minHandDetectionConfidence: minDetectionConfidence,
                    minTrackingConfidence: minTrackingConfidence,
                });

                // Initialize Face Landmarker
                const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
                    baseOptions: {
                        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                        delegate: 'GPU',
                    },
                    runningMode: 'VIDEO',
                    numFaces: 1,
                    minFaceDetectionConfidence: minDetectionConfidence,
                    minTrackingConfidence: minTrackingConfidence,
                });

                if (mounted) {
                    poseLandmarkerRef.current = poseLandmarker;
                    handLandmarkerRef.current = handLandmarker;
                    faceLandmarkerRef.current = faceLandmarker;
                    setIsLoading(false);
                }
            } catch (err) {
                console.error('MediaPipe initialization error:', err);
                if (mounted) {
                    setError(err instanceof Error ? err.message : 'Failed to initialize MediaPipe');
                    setIsLoading(false);
                }
            }
        }

        init();

        return () => {
            mounted = false;
            poseLandmarkerRef.current?.close();
            handLandmarkerRef.current?.close();
            faceLandmarkerRef.current?.close();
        };
    }, [minDetectionConfidence, minTrackingConfidence]);

    const processFrame = useCallback(
        async (video: HTMLVideoElement, timestamp: number) => {
            if (
                !poseLandmarkerRef.current ||
                !handLandmarkerRef.current ||
                !faceLandmarkerRef.current ||
                isProcessingRef.current
            ) {
                return;
            }

            isProcessingRef.current = true;

            try {
                // Process all landmarks
                const poseResults = poseLandmarkerRef.current.detectForVideo(video, timestamp);
                const handResults = handLandmarkerRef.current.detectForVideo(video, timestamp);
                const faceResults = faceLandmarkerRef.current.detectForVideo(video, timestamp);

                // Extract pose landmarks (33 landmarks)
                const pose = poseResults.landmarks[0]
                    ? poseResults.landmarks[0].map((lm) => [lm.x, lm.y, lm.z, lm.visibility || 1])
                    : Array(33).fill([0, 0, 0, 0]);

                // Extract hand landmarks (21 each)
                const leftHand = handResults.landmarks[0]
                    ? handResults.landmarks[0].map((lm) => [lm.x, lm.y, lm.z])
                    : Array(21).fill([0, 0, 0]);

                const rightHand = handResults.landmarks[1]
                    ? handResults.landmarks[1].map((lm) => [lm.x, lm.y, lm.z])
                    : Array(21).fill([0, 0, 0]);

                // Extract essential face landmarks
                const face = faceResults.faceLandmarks && faceResults.faceLandmarks[0]
                    ? extractEssentialFaceLandmarks(faceResults.faceLandmarks[0])
                    : Array(50).fill([0, 0, 0]);

                // Calculate confidence
                const poseConfidence = poseResults.landmarks[0]
                    ? poseResults.landmarks[0].reduce((acc, lm) => acc + (lm.visibility || 0), 0) / 33
                    : 0;
                const handConfidence = (handResults.landmarks.length > 0 ? 1 : 0);
                const confidence = (poseConfidence + handConfidence) / 2;

                const landmarks: LandmarksData = {
                    pose,
                    leftHand,
                    rightHand,
                    face,
                    // Keep raw coordinates for drawing (before normalization)
                    rawLeftHand: leftHand.map(pt => [...pt]),
                    rawRightHand: rightHand.map(pt => [...pt]),
                    confidence,
                    timestamp,
                };

                // Normalize landmarks for ML training
                const normalized = normalizeLandmarks(landmarks);

                if (onLandmarks) {
                    onLandmarks(normalized);
                }
            } catch (err) {
                console.error('Frame processing error:', err);
            } finally {
                isProcessingRef.current = false;
            }
        },
        [onLandmarks]
    );

    return {
        isLoading,
        error,
        isProcessing: isProcessingRef.current,
        processFrame,
    };
}

// Helper: Extract essential facial landmarks (50 key points)
function extractEssentialFaceLandmarks(allLandmarks: any[]): number[][] {
    const essentialIndices = [
        // Left eye
        33, 133, 160, 159, 158, 157, 173, 144, 145, 153,
        // Right eye  
        362, 263, 387, 386, 385, 384, 398, 373, 374, 380,
        // Left eyebrow
        70, 63, 105, 66, 107,
        // Right eyebrow
        300, 293, 334, 296, 336,
        // Mouth
        61, 291, 0, 17, 269, 39, 37, 40, 185, 146, 91, 181, 84, 314, 405, 321, 375,
    ];

    return essentialIndices.map((idx) => {
        const lm = allLandmarks[idx] || { x: 0, y: 0, z: 0 };
        return [lm.x, lm.y, lm.z];
    });
}

// Helper: Normalize landmarks relative to shoulder midpoint
function normalizeLandmarks(landmarks: LandmarksData): LandmarksData {
    const { pose } = landmarks;

    // Get shoulder landmarks (11 = left shoulder, 12 = right shoulder)
    const leftShoulder = pose[11];
    const rightShoulder = pose[12];

    if (!leftShoulder || !rightShoulder) {
        return landmarks;
    }

    // Calculate anchor (shoulder midpoint)
    const anchor = [
        (leftShoulder[0] + rightShoulder[0]) / 2,
        (leftShoulder[1] + rightShoulder[1]) / 2,
        (leftShoulder[2] + rightShoulder[2]) / 2,
    ];

    // Calculate scale (shoulder width)
    const scale = Math.sqrt(
        Math.pow(leftShoulder[0] - rightShoulder[0], 2) +
        Math.pow(leftShoulder[1] - rightShoulder[1], 2) +
        Math.pow(leftShoulder[2] - rightShoulder[2], 2)
    );

    if (scale < 0.01) {
        return landmarks;
    }

    // Normalize all landmarks
    const normalizePoint = (point: number[]) => [
        (point[0] - anchor[0]) / scale,
        (point[1] - anchor[1]) / scale,
        (point[2] - anchor[2]) / scale,
        ...(point.length > 3 ? [point[3]] : []),
    ];

    return {
        ...landmarks,
        pose: pose.map(normalizePoint),
        leftHand: landmarks.leftHand.map(normalizePoint),
        rightHand: landmarks.rightHand.map(normalizePoint),
        face: landmarks.face.map(normalizePoint),
    };
}
