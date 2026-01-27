/**
 * Text to ASL Gloss Conversion
 * Converts English sentences to ASL gloss notation
 * 
 * ASL Grammar Rules:
 * - Topic-Comment structure (object first)
 * - No articles (a, an, the)
 * - No "be" verbs in many cases
 * - Time markers come first
 * - Questions use non-manual markers
 */

// Common words to remove (not used in ASL)
const STOP_WORDS = new Set([
    'a', 'an', 'the', 'is', 'are', 'am', 'was', 'were', 'be', 'been', 'being',
    'do', 'does', 'did', 'have', 'has', 'had', 'to', 'of', 'for', 'with',
    'at', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 'just', 'also'
]);

// Time markers that should come first in ASL
const TIME_MARKERS = new Set([
    'yesterday', 'today', 'tomorrow', 'now', 'later', 'before', 'after',
    'morning', 'afternoon', 'evening', 'night', 'week', 'month', 'year',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'
]);

// Pronoun mapping
const PRONOUN_MAP: Record<string, string> = {
    'i': 'I',
    'me': 'I',
    'my': 'I',
    'mine': 'I',
    'you': 'YOU',
    'your': 'YOU',
    'yours': 'YOU',
    'he': 'HE',
    'him': 'HE',
    'his': 'HE',
    'she': 'SHE',
    'her': 'SHE',
    'hers': 'SHE',
    'it': 'IT',
    'its': 'IT',
    'we': 'WE',
    'us': 'WE',
    'our': 'WE',
    'ours': 'WE',
    'they': 'THEY',
    'them': 'THEY',
    'their': 'THEY',
    'theirs': 'THEY'
};

// Common contractions
const CONTRACTIONS: Record<string, string[]> = {
    "i'm": ['i', 'am'],
    "you're": ['you', 'are'],
    "he's": ['he', 'is'],
    "she's": ['she', 'is'],
    "it's": ['it', 'is'],
    "we're": ['we', 'are'],
    "they're": ['they', 'are'],
    "i've": ['i', 'have'],
    "you've": ['you', 'have'],
    "we've": ['we', 'have'],
    "they've": ['they', 'have'],
    "i'll": ['i', 'will'],
    "you'll": ['you', 'will'],
    "he'll": ['he', 'will'],
    "she'll": ['she', 'will'],
    "we'll": ['we', 'will'],
    "they'll": ['they', 'will'],
    "isn't": ['is', 'not'],
    "aren't": ['are', 'not'],
    "wasn't": ['was', 'not'],
    "weren't": ['were', 'not'],
    "don't": ['do', 'not'],
    "doesn't": ['does', 'not'],
    "didn't": ['did', 'not'],
    "won't": ['will', 'not'],
    "can't": ['can', 'not'],
    "couldn't": ['could', 'not'],
    "shouldn't": ['should', 'not'],
    "wouldn't": ['would', 'not']
};

/**
 * Expand contractions in text
 */
function expandContractions(text: string): string {
    let result = text.toLowerCase();
    for (const [contraction, words] of Object.entries(CONTRACTIONS)) {
        result = result.replace(new RegExp(contraction, 'gi'), words.join(' '));
    }
    return result;
}

/**
 * Tokenize text into words
 */
function tokenize(text: string): string[] {
    return text
        .toLowerCase()
        .replace(/[^\w\s']/g, ' ')
        .split(/\s+/)
        .filter(word => word.length > 0);
}

/**
 * Convert text to ASL Gloss
 * @param text - English text to convert
 * @returns Array of gloss tokens
 */
export function textToGloss(text: string): string[] {
    // Expand contractions
    const expanded = expandContractions(text);

    // Tokenize
    const tokens = tokenize(expanded);

    // Separate time markers and other words
    const timeMarkers: string[] = [];
    const otherWords: string[] = [];

    for (const token of tokens) {
        if (TIME_MARKERS.has(token)) {
            timeMarkers.push(token.toUpperCase());
        } else if (!STOP_WORDS.has(token)) {
            // Map pronouns
            const mapped = PRONOUN_MAP[token] || token.toUpperCase();
            otherWords.push(mapped);
        }
    }

    // ASL structure: TIME + TOPIC + COMMENT
    // For now, simple reordering: time first, then other words
    const gloss = [...timeMarkers, ...otherWords];

    // Remove duplicates while preserving order
    const seen = new Set<string>();
    return gloss.filter(word => {
        if (seen.has(word)) return false;
        seen.add(word);
        return true;
    });
}

/**
 * Convert gloss array to display string
 */
export function glossToString(gloss: string[]): string {
    return gloss.join(' ');
}

/**
 * Check if text is a question
 */
export function isQuestion(text: string): boolean {
    const trimmed = text.trim();
    return trimmed.endsWith('?') ||
        /^(who|what|when|where|why|how|which|whose|whom|do|does|did|is|are|was|were|can|could|will|would|should|have|has|had)\b/i.test(trimmed);
}

/**
 * Get question type for non-manual markers
 */
export function getQuestionType(text: string): 'wh' | 'yes-no' | 'none' {
    const trimmed = text.trim().toLowerCase();

    if (/^(who|what|when|where|why|how|which|whose|whom)\b/.test(trimmed)) {
        return 'wh';
    }

    if (trimmed.endsWith('?') || /^(do|does|did|is|are|was|were|can|could|will|would|should|have|has|had)\b/.test(trimmed)) {
        return 'yes-no';
    }

    return 'none';
}

// Export types
export interface GlossResult {
    gloss: string[];
    isQuestion: boolean;
    questionType: 'wh' | 'yes-no' | 'none';
    original: string;
}

/**
 * Full text to gloss conversion with metadata
 */
export function convertToGloss(text: string): GlossResult {
    return {
        gloss: textToGloss(text),
        isQuestion: isQuestion(text),
        questionType: getQuestionType(text),
        original: text
    };
}
