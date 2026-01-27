/**
 * Sentiment Analysis for Emotion Detection
 * Simple rule-based sentiment analysis for avatar facial expressions
 */

export type Emotion = 'neutral' | 'happy' | 'sad' | 'angry' | 'surprised' | 'confused';

// Emotion keyword mappings
const EMOTION_KEYWORDS: Record<Emotion, string[]> = {
    happy: [
        'happy', 'joy', 'joyful', 'glad', 'pleased', 'delighted', 'excited',
        'wonderful', 'amazing', 'great', 'awesome', 'love', 'loving', 'loved',
        'like', 'enjoy', 'fun', 'funny', 'laugh', 'smile', 'good', 'nice',
        'beautiful', 'fantastic', 'excellent', 'perfect', 'best', 'thank',
        'thanks', 'grateful', 'appreciate', 'celebrate', 'congratulations',
        'yay', 'hooray', 'wow', 'cool', 'sweet', 'blessed'
    ],
    sad: [
        'sad', 'unhappy', 'depressed', 'down', 'upset', 'disappointed',
        'sorry', 'regret', 'miss', 'missing', 'lonely', 'alone', 'hurt',
        'pain', 'painful', 'cry', 'crying', 'tears', 'heartbroken', 'broken',
        'lost', 'losing', 'fail', 'failed', 'failure', 'bad', 'terrible',
        'awful', 'horrible', 'worst', 'unfortunately', 'tragic', 'grief'
    ],
    angry: [
        'angry', 'mad', 'furious', 'rage', 'hate', 'hating', 'hated',
        'annoyed', 'annoying', 'irritated', 'frustrated', 'frustrating',
        'upset', 'outraged', 'disgusted', 'disgusting', 'sick', 'enough',
        'stop', 'stupid', 'idiot', 'fool', 'damn', 'hell', 'shut'
    ],
    surprised: [
        'surprised', 'surprising', 'shock', 'shocked', 'shocking', 'wow',
        'whoa', 'omg', 'unbelievable', 'incredible', 'amazing', 'unexpected',
        'sudden', 'suddenly', 'really', 'seriously', 'what', 'how'
    ],
    confused: [
        'confused', 'confusing', 'confusion', 'puzzled', 'puzzling',
        'uncertain', 'unsure', 'unclear', 'strange', 'weird', 'odd',
        'wonder', 'wondering', 'maybe', 'perhaps', 'probably', 'huh',
        'what', 'why', 'how', 'understand', 'explain'
    ],
    neutral: []
};

// Negation words that can flip sentiment
const NEGATION_WORDS = new Set([
    'not', 'no', 'never', 'none', 'nothing', 'nobody', 'nowhere',
    'neither', 'hardly', 'barely', 'scarcely', "don't", "doesn't",
    "didn't", "won't", "wouldn't", "couldn't", "shouldn't", "isn't",
    "aren't", "wasn't", "weren't", "can't"
]);

// Intensity modifiers
const INTENSIFIERS = new Set([
    'very', 'really', 'extremely', 'absolutely', 'completely', 'totally',
    'so', 'such', 'quite', 'rather', 'pretty', 'too', 'super', 'incredibly'
]);

/**
 * Tokenize text for analysis
 */
function tokenize(text: string): string[] {
    return text
        .toLowerCase()
        .replace(/[^\w\s']/g, ' ')
        .split(/\s+/)
        .filter(word => word.length > 0);
}

/**
 * Count emotion matches in text
 */
function countEmotionMatches(tokens: string[]): Record<Emotion, number> {
    const counts: Record<Emotion, number> = {
        neutral: 0,
        happy: 0,
        sad: 0,
        angry: 0,
        surprised: 0,
        confused: 0
    };

    let hasNegation = false;
    let hasIntensifier = false;

    for (let i = 0; i < tokens.length; i++) {
        const token = tokens[i];

        // Check for negation (affects next word)
        if (NEGATION_WORDS.has(token)) {
            hasNegation = true;
            continue;
        }

        // Check for intensifiers
        if (INTENSIFIERS.has(token)) {
            hasIntensifier = true;
            continue;
        }

        // Check each emotion category
        for (const [emotion, keywords] of Object.entries(EMOTION_KEYWORDS)) {
            if (keywords.includes(token)) {
                const multiplier = hasIntensifier ? 2 : 1;

                // Negation flips happy/sad
                if (hasNegation) {
                    if (emotion === 'happy') {
                        counts.sad += multiplier;
                    } else if (emotion === 'sad') {
                        counts.happy += multiplier;
                    } else {
                        counts[emotion as Emotion] += multiplier;
                    }
                } else {
                    counts[emotion as Emotion] += multiplier;
                }
            }
        }

        // Reset modifiers after processing a word
        hasNegation = false;
        hasIntensifier = false;
    }

    return counts;
}

/**
 * Detect the primary emotion in text
 */
export function detectEmotion(text: string): Emotion {
    const tokens = tokenize(text);
    const counts = countEmotionMatches(tokens);

    // Find the emotion with highest count
    let maxEmotion: Emotion = 'neutral';
    let maxCount = 0;

    for (const [emotion, count] of Object.entries(counts)) {
        if (count > maxCount && emotion !== 'neutral') {
            maxCount = count;
            maxEmotion = emotion as Emotion;
        }
    }

    return maxEmotion;
}

/**
 * Get emotion confidence score (0-1)
 */
export function getEmotionConfidence(text: string): { emotion: Emotion; confidence: number } {
    const tokens = tokenize(text);
    const counts = countEmotionMatches(tokens);

    const totalMatches = Object.values(counts).reduce((a, b) => a + b, 0);

    if (totalMatches === 0) {
        return { emotion: 'neutral', confidence: 1.0 };
    }

    let maxEmotion: Emotion = 'neutral';
    let maxCount = 0;

    for (const [emotion, count] of Object.entries(counts)) {
        if (count > maxCount && emotion !== 'neutral') {
            maxCount = count;
            maxEmotion = emotion as Emotion;
        }
    }

    return {
        emotion: maxEmotion,
        confidence: maxCount / Math.max(totalMatches, tokens.length * 0.3)
    };
}

/**
 * Get all detected emotions with scores
 */
export function analyzeEmotions(text: string): Record<Emotion, number> {
    const tokens = tokenize(text);
    const counts = countEmotionMatches(tokens);

    // Normalize to 0-1 range
    const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;

    return {
        neutral: counts.neutral / total || (total === 0 ? 1 : 0),
        happy: counts.happy / total,
        sad: counts.sad / total,
        angry: counts.angry / total,
        surprised: counts.surprised / total,
        confused: counts.confused / total
    };
}
