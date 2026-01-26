'use client';

// Canvas drawing utilities for skeleton visualization

// MediaPipe connection pairs for skeleton
export const POSE_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 7], // Right eye
    [0, 4], [4, 5], [5, 6], [6, 8], // Left eye
    [9, 10], // Mouth
    [11, 12], // Shoulders
    [11, 13], [13, 15], // Left arm
    [12, 14], [14, 16], // Right arm
    [11, 23], [12, 24], // Torso
    [23, 24], // Hips
    [23, 25], [25, 27], // Left leg
    [24, 26], [26, 28], // Right leg
];

export const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8], // Index
    [0, 9], [9, 10], [10, 11], [11, 12], // Middle
    [0, 13], [13, 14], [14, 15], [15, 16], // Ring
    [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
    [5, 9], [9, 13], [13, 17], // Palm
];

interface DrawOptions {
    color?: string;
    lineWidth?: number;
    pointRadius?: number;
}

export function drawLandmarks(
    ctx: CanvasRenderingContext2D,
    landmarks: number[][],
    width: number,
    height: number,
    options: DrawOptions = {}
) {
    const { color = '#00ff00', pointRadius = 3 } = options;

    ctx.fillStyle = color;

    for (const lm of landmarks) {
        if (lm[0] === 0 && lm[1] === 0 && lm[2] === 0) continue; // Skip empty

        const x = lm[0] * width;
        const y = lm[1] * height;

        ctx.beginPath();
        ctx.arc(x, y, pointRadius, 0, 2 * Math.PI);
        ctx.fill();
    }
}

export function drawConnections(
    ctx: CanvasRenderingContext2D,
    landmarks: number[][],
    connections: number[][],
    width: number,
    height: number,
    options: DrawOptions = {}
) {
    const { color = '#00ff00', lineWidth = 2 } = options;

    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;

    for (const [i, j] of connections) {
        if (i >= landmarks.length || j >= landmarks.length) continue;

        const start = landmarks[i];
        const end = landmarks[j];

        // Skip if either point is empty
        if ((start[0] === 0 && start[1] === 0) || (end[0] === 0 && end[1] === 0)) continue;

        ctx.beginPath();
        ctx.moveTo(start[0] * width, start[1] * height);
        ctx.lineTo(end[0] * width, end[1] * height);
        ctx.stroke();
    }
}

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

    // Draw pose (green)
    drawConnections(ctx, data.pose, POSE_CONNECTIONS, width, height, { color: '#00ff00', lineWidth: 3 });
    drawLandmarks(ctx, data.pose, width, height, { color: '#00ff00', pointRadius: 4 });

    // Draw left hand (blue)
    drawConnections(ctx, data.leftHand, HAND_CONNECTIONS, width, height, { color: '#00ccff', lineWidth: 2 });
    drawLandmarks(ctx, data.leftHand, width, height, { color: '#00ccff', pointRadius: 3 });

    // Draw right hand (orange)
    drawConnections(ctx, data.rightHand, HAND_CONNECTIONS, width, height, { color: '#ff9900', lineWidth: 2 });
    drawLandmarks(ctx, data.rightHand, width, height, { color: '#ff9900', pointRadius: 3 });

    // Draw face points (pink)
    drawLandmarks(ctx, data.face, width, height, { color: '#ff66cc', pointRadius: 2 });
}
