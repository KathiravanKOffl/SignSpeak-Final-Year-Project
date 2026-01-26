'use client';

// Refined skeleton drawing utilities

// Pose connections (main body)
export const POSE_CONNECTIONS = [
    // Face outline
    [0, 1], [1, 2], [2, 3], [3, 7],  // Right eyebrow
    [0, 4], [4, 5], [5, 6], [6, 8],  // Left eyebrow
    [9, 10], // Mouth
    // Torso
    [11, 12], // Shoulders
    [11, 23], [12, 24], // Shoulder to hip
    [23, 24], // Hips
    // Arms
    [11, 13], [13, 15], // Left arm
    [12, 14], [14, 16], // Right arm
    // Legs (optional - uncomment if needed)
    // [23, 25], [25, 27], // Left leg
    // [24, 26], [26, 28], // Right leg
];

// Hand connections
export const HAND_CONNECTIONS = [
    // Thumb
    [0, 1], [1, 2], [2, 3], [3, 4],
    // Index
    [0, 5], [5, 6], [6, 7], [7, 8],
    // Middle
    [0, 9], [9, 10], [10, 11], [11, 12],
    // Ring
    [0, 13], [13, 14], [14, 15], [15, 16],
    // Pinky
    [0, 17], [17, 18], [18, 19], [19, 20],
    // Palm
    [5, 9], [9, 13], [13, 17],
];

// Check if landmark is valid (not zero/empty)
function isValidLandmark(lm: number[]): boolean {
    return lm && (Math.abs(lm[0]) > 0.001 || Math.abs(lm[1]) > 0.001);
}

// Draw smooth line between two points with gradient
function drawSmoothLine(
    ctx: CanvasRenderingContext2D,
    x1: number, y1: number,
    x2: number, y2: number,
    color: string,
    lineWidth: number = 2
) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
}

// Draw a glowing point
function drawGlowPoint(
    ctx: CanvasRenderingContext2D,
    x: number, y: number,
    radius: number,
    color: string
) {
    // Outer glow
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius * 2);
    gradient.addColorStop(0, color);
    gradient.addColorStop(0.5, color + '88');
    gradient.addColorStop(1, 'transparent');

    ctx.beginPath();
    ctx.fillStyle = gradient;
    ctx.arc(x, y, radius * 2, 0, Math.PI * 2);
    ctx.fill();

    // Inner solid point
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
}

// Draw hand skeleton
function drawHand(
    ctx: CanvasRenderingContext2D,
    landmarks: number[][],
    width: number,
    height: number,
    color: string
) {
    // Check if hand is detected (at least some valid landmarks)
    const validCount = landmarks.filter(isValidLandmark).length;
    if (validCount < 5) return;

    // Draw connections first (behind points)
    ctx.globalAlpha = 0.8;
    for (const [i, j] of HAND_CONNECTIONS) {
        if (i >= landmarks.length || j >= landmarks.length) continue;

        const start = landmarks[i];
        const end = landmarks[j];

        if (!isValidLandmark(start) || !isValidLandmark(end)) continue;

        const x1 = start[0] * width;
        const y1 = start[1] * height;
        const x2 = end[0] * width;
        const y2 = end[1] * height;

        drawSmoothLine(ctx, x1, y1, x2, y2, color, 3);
    }

    // Draw points on top
    ctx.globalAlpha = 1;
    for (let i = 0; i < landmarks.length; i++) {
        const lm = landmarks[i];
        if (!isValidLandmark(lm)) continue;

        const x = lm[0] * width;
        const y = lm[1] * height;

        // Highlight fingertips
        const isFingertip = [4, 8, 12, 16, 20].includes(i);
        drawGlowPoint(ctx, x, y, isFingertip ? 5 : 3, color);
    }
}

// Draw pose skeleton
function drawPose(
    ctx: CanvasRenderingContext2D,
    landmarks: number[][],
    width: number,
    height: number
) {
    const validCount = landmarks.filter(isValidLandmark).length;
    if (validCount < 5) return;

    const color = '#22c55e'; // Green

    // Draw connections
    ctx.globalAlpha = 0.6;
    for (const [i, j] of POSE_CONNECTIONS) {
        if (i >= landmarks.length || j >= landmarks.length) continue;

        const start = landmarks[i];
        const end = landmarks[j];

        if (!isValidLandmark(start) || !isValidLandmark(end)) continue;

        const x1 = start[0] * width;
        const y1 = start[1] * height;
        const x2 = end[0] * width;
        const y2 = end[1] * height;

        drawSmoothLine(ctx, x1, y1, x2, y2, color, 2);
    }

    // Draw key points only (shoulders, elbows, wrists)
    ctx.globalAlpha = 1;
    const keyPoints = [11, 12, 13, 14, 15, 16]; // Shoulders, elbows, wrists
    for (const i of keyPoints) {
        if (i >= landmarks.length) continue;
        const lm = landmarks[i];
        if (!isValidLandmark(lm)) continue;

        const x = lm[0] * width;
        const y = lm[1] * height;
        drawGlowPoint(ctx, x, y, 4, color);
    }
}

// Main skeleton drawing function
export function drawSkeleton(
    ctx: CanvasRenderingContext2D,
    data: {
        pose: number[][];
        leftHand: number[][];
        rightHand: number[][];
        face: number[][];
    },
    width: number,
    height: number
) {
    ctx.clearRect(0, 0, width, height);

    // Draw pose (subtle green)
    drawPose(ctx, data.pose, width, height);

    // Draw hands (more prominent)
    drawHand(ctx, data.leftHand, width, height, '#3b82f6'); // Blue
    drawHand(ctx, data.rightHand, width, height, '#f97316'); // Orange

    // Skip face drawing (too cluttered)
    ctx.globalAlpha = 1;
}
