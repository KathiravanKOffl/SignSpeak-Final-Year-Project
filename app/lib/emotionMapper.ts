/**
 * Emotion to Avatar Expression Mapper
 * Maps detected emotions to avatar facial blendshapes and body language
 */

import type { Emotion } from './sentimentAnalysis';

// Facial blendshape values (0-1 range)
export interface FacialExpression {
    // Eyebrows
    browInnerUp: number;
    browOuterUp: number;
    browDown: number;

    // Eyes
    eyeSquint: number;
    eyeWide: number;
    eyeLookUp: number;
    eyeLookDown: number;

    // Mouth
    mouthSmile: number;
    mouthFrown: number;
    mouthOpen: number;
    mouthPucker: number;
    jawOpen: number;

    // Cheeks
    cheekPuff: number;
    cheekSquint: number;
}

// Default neutral expression
const NEUTRAL_EXPRESSION: FacialExpression = {
    browInnerUp: 0,
    browOuterUp: 0,
    browDown: 0,
    eyeSquint: 0,
    eyeWide: 0,
    eyeLookUp: 0,
    eyeLookDown: 0,
    mouthSmile: 0,
    mouthFrown: 0,
    mouthOpen: 0,
    mouthPucker: 0,
    jawOpen: 0,
    cheekPuff: 0,
    cheekSquint: 0
};

// Emotion expression presets
const EMOTION_EXPRESSIONS: Record<Emotion, FacialExpression> = {
    neutral: NEUTRAL_EXPRESSION,

    happy: {
        browInnerUp: 0.2,
        browOuterUp: 0.3,
        browDown: 0,
        eyeSquint: 0.4,
        eyeWide: 0,
        eyeLookUp: 0.1,
        eyeLookDown: 0,
        mouthSmile: 0.8,
        mouthFrown: 0,
        mouthOpen: 0.2,
        mouthPucker: 0,
        jawOpen: 0.1,
        cheekPuff: 0,
        cheekSquint: 0.5
    },

    sad: {
        browInnerUp: 0.6,
        browOuterUp: 0,
        browDown: 0.3,
        eyeSquint: 0.2,
        eyeWide: 0,
        eyeLookUp: 0,
        eyeLookDown: 0.4,
        mouthSmile: 0,
        mouthFrown: 0.7,
        mouthOpen: 0.1,
        mouthPucker: 0.2,
        jawOpen: 0,
        cheekPuff: 0,
        cheekSquint: 0
    },

    angry: {
        browInnerUp: 0,
        browOuterUp: 0,
        browDown: 0.8,
        eyeSquint: 0.5,
        eyeWide: 0.2,
        eyeLookUp: 0,
        eyeLookDown: 0,
        mouthSmile: 0,
        mouthFrown: 0.6,
        mouthOpen: 0.3,
        mouthPucker: 0,
        jawOpen: 0.2,
        cheekPuff: 0.1,
        cheekSquint: 0.3
    },

    surprised: {
        browInnerUp: 0.7,
        browOuterUp: 0.8,
        browDown: 0,
        eyeSquint: 0,
        eyeWide: 0.9,
        eyeLookUp: 0.2,
        eyeLookDown: 0,
        mouthSmile: 0,
        mouthFrown: 0,
        mouthOpen: 0.6,
        mouthPucker: 0,
        jawOpen: 0.5,
        cheekPuff: 0,
        cheekSquint: 0
    },

    confused: {
        browInnerUp: 0.4,
        browOuterUp: 0.2,
        browDown: 0.2,
        eyeSquint: 0.3,
        eyeWide: 0,
        eyeLookUp: 0.3,
        eyeLookDown: 0,
        mouthSmile: 0,
        mouthFrown: 0.2,
        mouthOpen: 0.1,
        mouthPucker: 0.3,
        jawOpen: 0,
        cheekPuff: 0,
        cheekSquint: 0
    }
};

/**
 * Get facial expression for an emotion
 */
export function getExpression(emotion: Emotion): FacialExpression {
    return EMOTION_EXPRESSIONS[emotion] || NEUTRAL_EXPRESSION;
}

/**
 * Blend two expressions together
 * @param from - Starting expression
 * @param to - Target expression
 * @param t - Blend factor (0-1)
 */
export function blendExpressions(
    from: FacialExpression,
    to: FacialExpression,
    t: number
): FacialExpression {
    const result: Partial<FacialExpression> = {};

    for (const key of Object.keys(from) as (keyof FacialExpression)[]) {
        result[key] = from[key] * (1 - t) + to[key] * t;
    }

    return result as FacialExpression;
}

/**
 * Get expression for question types (non-manual markers in ASL)
 */
export function getQuestionExpression(type: 'wh' | 'yes-no' | 'none'): Partial<FacialExpression> {
    switch (type) {
        case 'wh':
            // WH-questions: furrowed brow, head tilted
            return {
                browDown: 0.5,
                browInnerUp: 0.3,
                eyeSquint: 0.2
            };
        case 'yes-no':
            // Yes/No questions: raised eyebrows, widened eyes
            return {
                browInnerUp: 0.6,
                browOuterUp: 0.7,
                eyeWide: 0.4
            };
        default:
            return {};
    }
}

/**
 * Body language parameters
 */
export interface BodyLanguage {
    headTilt: number;      // -1 to 1 (left to right)
    headNod: number;       // -1 to 1 (down to up)
    shoulderRaise: number; // 0 to 1
    leanForward: number;   // 0 to 1
}

/**
 * Get body language for emotion
 */
export function getBodyLanguage(emotion: Emotion): BodyLanguage {
    const bodyMap: Record<Emotion, BodyLanguage> = {
        neutral: { headTilt: 0, headNod: 0, shoulderRaise: 0, leanForward: 0 },
        happy: { headTilt: 0.1, headNod: 0.2, shoulderRaise: 0, leanForward: 0.1 },
        sad: { headTilt: -0.1, headNod: -0.3, shoulderRaise: 0.2, leanForward: -0.1 },
        angry: { headTilt: 0, headNod: 0.1, shoulderRaise: 0.3, leanForward: 0.3 },
        surprised: { headTilt: 0, headNod: 0.2, shoulderRaise: 0.4, leanForward: -0.2 },
        confused: { headTilt: 0.3, headNod: 0, shoulderRaise: 0.1, leanForward: 0 }
    };

    return bodyMap[emotion] || bodyMap.neutral;
}

/**
 * Smooth transition helper
 */
export function easeInOutCubic(t: number): number {
    return t < 0.5
        ? 4 * t * t * t
        : 1 - Math.pow(-2 * t + 2, 3) / 2;
}
