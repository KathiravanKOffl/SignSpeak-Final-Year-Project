/**
 * Sign Animation Data and Helpers
 * Maps ASL glosses to animation keyframes for the avatar
 */

// Joint rotation in degrees (pitch, yaw, roll)
export interface JointRotation {
    x: number;
    y: number;
    z: number;
}

// Hand pose configuration
export interface HandPose {
    // Finger curl values (0 = extended, 1 = fully curled)
    thumb: number;
    index: number;
    middle: number;
    ring: number;
    pinky: number;

    // Hand orientation
    palmDirection: 'forward' | 'back' | 'up' | 'down' | 'left' | 'right';

    // Finger spread
    spread: number;
}

// Arm configuration
export interface ArmPose {
    shoulder: JointRotation;
    elbow: JointRotation;
    wrist: JointRotation;
}

// Full sign keyframe
export interface SignKeyframe {
    time: number; // 0-1 through the sign
    leftHand: HandPose;
    rightHand: HandPose;
    leftArm: ArmPose;
    rightArm: ArmPose;
}

// Complete sign animation
export interface SignAnimation {
    gloss: string;
    duration: number; // milliseconds
    keyframes: SignKeyframe[];
    handshape: 'one-hand' | 'two-hand' | 'symmetric';
    movement: 'static' | 'linear' | 'circular' | 'arc';
}

// Default hand poses
const HAND_FLAT: HandPose = {
    thumb: 0.2, index: 0, middle: 0, ring: 0, pinky: 0,
    palmDirection: 'forward', spread: 0.3
};

const HAND_FIST: HandPose = {
    thumb: 0.8, index: 1, middle: 1, ring: 1, pinky: 1,
    palmDirection: 'forward', spread: 0
};

const HAND_POINT: HandPose = {
    thumb: 0.8, index: 0, middle: 1, ring: 1, pinky: 1,
    palmDirection: 'forward', spread: 0
};

// Default arm poses
const ARM_REST: ArmPose = {
    shoulder: { x: 0, y: 0, z: 0 },
    elbow: { x: 0, y: 0, z: 0 },
    wrist: { x: 0, y: 0, z: 0 }
};

const ARM_RAISED: ArmPose = {
    shoulder: { x: -60, y: 0, z: 0 },
    elbow: { x: -30, y: 0, z: 0 },
    wrist: { x: 0, y: 0, z: 0 }
};

const ARM_CHEST: ArmPose = {
    shoulder: { x: -30, y: 30, z: 0 },
    elbow: { x: -90, y: 0, z: 0 },
    wrist: { x: 0, y: 0, z: 0 }
};

// ASL Alphabet animations
const ASL_ALPHABET: Record<string, SignKeyframe[]> = {
    'A': [{ time: 0, leftHand: HAND_FLAT, rightHand: { ...HAND_FIST, thumb: 0 }, leftArm: ARM_REST, rightArm: ARM_CHEST }],
    'B': [{ time: 0, leftHand: HAND_FLAT, rightHand: { ...HAND_FLAT, thumb: 1 }, leftArm: ARM_REST, rightArm: ARM_RAISED }],
    'C': [{ time: 0, leftHand: HAND_FLAT, rightHand: { thumb: 0.3, index: 0.3, middle: 0.3, ring: 0.3, pinky: 0.3, palmDirection: 'left', spread: 0.2 }, leftArm: ARM_REST, rightArm: ARM_CHEST }],
    'D': [{ time: 0, leftHand: HAND_FLAT, rightHand: { thumb: 0.5, index: 0, middle: 0.8, ring: 0.8, pinky: 0.8, palmDirection: 'forward', spread: 0 }, leftArm: ARM_REST, rightArm: ARM_RAISED }],
    'E': [{ time: 0, leftHand: HAND_FLAT, rightHand: { thumb: 0.5, index: 0.7, middle: 0.7, ring: 0.7, pinky: 0.7, palmDirection: 'forward', spread: 0 }, leftArm: ARM_REST, rightArm: ARM_CHEST }],
    // More letters would be added here...
};

// Common word signs
const ASL_WORDS: Record<string, SignAnimation> = {
    'HELLO': {
        gloss: 'HELLO',
        duration: 800,
        handshape: 'one-hand',
        movement: 'arc',
        keyframes: [
            { time: 0, leftHand: HAND_FLAT, rightHand: HAND_FLAT, leftArm: ARM_REST, rightArm: { shoulder: { x: -45, y: 45, z: 0 }, elbow: { x: -30, y: 0, z: 0 }, wrist: { x: 0, y: 0, z: 0 } } },
            { time: 0.5, leftHand: HAND_FLAT, rightHand: HAND_FLAT, leftArm: ARM_REST, rightArm: { shoulder: { x: -45, y: 90, z: 0 }, elbow: { x: -30, y: 0, z: 0 }, wrist: { x: 0, y: 0, z: 0 } } },
            { time: 1, leftHand: HAND_FLAT, rightHand: HAND_FLAT, leftArm: ARM_REST, rightArm: ARM_REST }
        ]
    },
    'THANK-YOU': {
        gloss: 'THANK-YOU',
        duration: 600,
        handshape: 'one-hand',
        movement: 'linear',
        keyframes: [
            { time: 0, leftHand: HAND_FLAT, rightHand: HAND_FLAT, leftArm: ARM_REST, rightArm: { shoulder: { x: -30, y: 0, z: 0 }, elbow: { x: -45, y: 30, z: 0 }, wrist: { x: -20, y: 0, z: 0 } } },
            { time: 1, leftHand: HAND_FLAT, rightHand: HAND_FLAT, leftArm: ARM_REST, rightArm: { shoulder: { x: -60, y: 0, z: 0 }, elbow: { x: -30, y: 60, z: 0 }, wrist: { x: 0, y: 0, z: 0 } } }
        ]
    },
    'I': {
        gloss: 'I',
        duration: 400,
        handshape: 'one-hand',
        movement: 'static',
        keyframes: [
            { time: 0, leftHand: HAND_FLAT, rightHand: HAND_POINT, leftArm: ARM_REST, rightArm: ARM_CHEST }
        ]
    },
    'YOU': {
        gloss: 'YOU',
        duration: 400,
        handshape: 'one-hand',
        movement: 'linear',
        keyframes: [
            { time: 0, leftHand: HAND_FLAT, rightHand: HAND_POINT, leftArm: ARM_REST, rightArm: ARM_CHEST },
            { time: 1, leftHand: HAND_FLAT, rightHand: HAND_POINT, leftArm: ARM_REST, rightArm: { shoulder: { x: -30, y: 60, z: 0 }, elbow: { x: -30, y: 0, z: 0 }, wrist: { x: 0, y: 0, z: 0 } } }
        ]
    },
    'LOVE': {
        gloss: 'LOVE',
        duration: 600,
        handshape: 'two-hand',
        movement: 'static',
        keyframes: [
            { time: 0, leftHand: HAND_FIST, rightHand: HAND_FIST, leftArm: { shoulder: { x: -20, y: 20, z: 0 }, elbow: { x: -90, y: 0, z: 0 }, wrist: { x: 0, y: 0, z: 0 } }, rightArm: { shoulder: { x: -20, y: -20, z: 0 }, elbow: { x: -90, y: 0, z: 0 }, wrist: { x: 0, y: 0, z: 0 } } }
        ]
    },
    'HELP': {
        gloss: 'HELP',
        duration: 700,
        handshape: 'two-hand',
        movement: 'linear',
        keyframes: [
            { time: 0, leftHand: HAND_FLAT, rightHand: { ...HAND_FIST, thumb: 0 }, leftArm: { shoulder: { x: -30, y: 20, z: 0 }, elbow: { x: -60, y: 0, z: 0 }, wrist: { x: 0, y: 0, z: 0 } }, rightArm: ARM_CHEST },
            { time: 1, leftHand: HAND_FLAT, rightHand: { ...HAND_FIST, thumb: 0 }, leftArm: { shoulder: { x: -60, y: 20, z: 0 }, elbow: { x: -30, y: 0, z: 0 }, wrist: { x: 0, y: 0, z: 0 } }, rightArm: ARM_RAISED }
        ]
    }
};

/**
 * Get animation for a gloss
 */
export function getSignAnimation(gloss: string): SignAnimation | null {
    const upperGloss = gloss.toUpperCase();

    // Check word dictionary first
    if (ASL_WORDS[upperGloss]) {
        return ASL_WORDS[upperGloss];
    }

    // Check if it's a single letter (fingerspelling)
    if (upperGloss.length === 1 && ASL_ALPHABET[upperGloss]) {
        return {
            gloss: upperGloss,
            duration: 500,
            handshape: 'one-hand',
            movement: 'static',
            keyframes: ASL_ALPHABET[upperGloss]
        };
    }

    return null;
}

/**
 * Get animation sequence for multiple glosses
 */
export function getAnimationSequence(glosses: string[]): { animation: SignAnimation; delay: number }[] {
    const sequence: { animation: SignAnimation; delay: number }[] = [];
    let currentDelay = 0;

    for (const gloss of glosses) {
        const animation = getSignAnimation(gloss);
        if (animation) {
            sequence.push({ animation, delay: currentDelay });
            currentDelay += animation.duration + 200; // 200ms pause between signs
        }
    }

    return sequence;
}

/**
 * Interpolate between keyframes using LERP
 */
export function interpolateKeyframes(
    from: SignKeyframe,
    to: SignKeyframe,
    t: number
): SignKeyframe {
    const lerp = (a: number, b: number) => a + (b - a) * t;

    const lerpJoint = (a: JointRotation, b: JointRotation): JointRotation => ({
        x: lerp(a.x, b.x),
        y: lerp(a.y, b.y),
        z: lerp(a.z, b.z)
    });

    const lerpHand = (a: HandPose, b: HandPose): HandPose => ({
        thumb: lerp(a.thumb, b.thumb),
        index: lerp(a.index, b.index),
        middle: lerp(a.middle, b.middle),
        ring: lerp(a.ring, b.ring),
        pinky: lerp(a.pinky, b.pinky),
        palmDirection: t < 0.5 ? a.palmDirection : b.palmDirection,
        spread: lerp(a.spread, b.spread)
    });

    const lerpArm = (a: ArmPose, b: ArmPose): ArmPose => ({
        shoulder: lerpJoint(a.shoulder, b.shoulder),
        elbow: lerpJoint(a.elbow, b.elbow),
        wrist: lerpJoint(a.wrist, b.wrist)
    });

    return {
        time: lerp(from.time, to.time),
        leftHand: lerpHand(from.leftHand, to.leftHand),
        rightHand: lerpHand(from.rightHand, to.rightHand),
        leftArm: lerpArm(from.leftArm, to.leftArm),
        rightArm: lerpArm(from.rightArm, to.rightArm)
    };
}

/**
 * Get current keyframe at a specific time in animation
 */
export function getKeyframeAtTime(animation: SignAnimation, time: number): SignKeyframe {
    const t = Math.min(1, Math.max(0, time / animation.duration));

    if (animation.keyframes.length === 1) {
        return animation.keyframes[0];
    }

    // Find surrounding keyframes
    let fromIndex = 0;
    let toIndex = 1;

    for (let i = 0; i < animation.keyframes.length - 1; i++) {
        if (animation.keyframes[i].time <= t && animation.keyframes[i + 1].time >= t) {
            fromIndex = i;
            toIndex = i + 1;
            break;
        }
    }

    const from = animation.keyframes[fromIndex];
    const to = animation.keyframes[toIndex];
    const localT = (t - from.time) / (to.time - from.time);

    return interpolateKeyframes(from, to, localT);
}
