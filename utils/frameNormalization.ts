/**
 * Frame Normalization Utilities
 * Ensures consistent temporal data collection across different camera FPS
 */

export interface NormalizedFrame {
    pose: number[][];
    leftHand: number[][];
    rightHand: number[][];
    face: number[][];
    timestamp: number;
}

/**
 * Resamples frame array to target count using linear interpolation
 * @param frames - Variable length frame array
 * @param targetCount - Desired frame count (e.g., 32)
 * @returns Exactly targetCount frames
 */
export function resampleToN(frames: any[], targetCount: number): any[] {
    if (frames.length === 0) return [];
    if (frames.length === targetCount) return frames;

    const step = (frames.length - 1) / (targetCount - 1);
    const resampled: any[] = [];

    for (let i = 0; i < targetCount; i++) {
        const idx = Math.round(i * step);
        resampled.push(frames[idx]);
    }

    return resampled;
}

/**
 * Flattens landmark data to a single array for storage
 * @param frame - Landmark frame data
 * @returns Flat array of numbers
 */
export function flattenFrame(frame: NormalizedFrame): number[] {
    const poseFlat = frame.pose.flatMap(lm => [lm[0], lm[1], lm[2]]);
    const leftHandFlat = frame.leftHand.flatMap(lm => lm);
    const rightHandFlat = frame.rightHand.flatMap(lm => lm);
    const faceFlat = frame.face.flatMap(lm => lm);

    return [...poseFlat, ...leftHandFlat, ...rightHandFlat, ...faceFlat];
}

/**
 * Validates that frame data is complete and correct
 */
export function validateFrameData(frames: any[]): boolean {
    if (!frames || frames.length === 0) return false;

    // Check each frame has required structure
    return frames.every(frame =>
        frame.pose &&
        frame.leftHand &&
        frame.rightHand &&
        frame.face
    );
}
