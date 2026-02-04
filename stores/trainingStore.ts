import { create } from 'zustand';

// Complete vocabulary (50 words across all tiers)
export const FULL_VOCABULARY = [
    // Tier 1: Grammar Core (10 words)
    'I', 'you', 'he', 'she', 'want', 'need', 'go', 'come', 'to', 'can',

    // Tier 2: Common Verbs (15 words)
    'eat', 'drink', 'help', 'sleep', 'work', 'study', 'play',
    'sit', 'stand', 'walk', 'run', 'stop', 'give', 'take', 'like',

    // Tier 3: Essential Nouns (15 words)
    'home', 'school', 'hospital', 'toilet', 'water', 'food',
    'phone', 'money', 'medicine', 'mother', 'father',
    'friend', 'doctor', 'teacher', 'child',

    // Tier 4: Modifiers (10 words)
    'good', 'bad', 'big', 'small', 'hot', 'cold',
    'now', 'later', 'today', 'tomorrow'
];

export interface TrainingSample {
    frames: any[]; // 32 normalized frames
    timestamp: number;
}

interface TrainingState {
    // Configuration
    currentWord: string;
    currentWordIndex: number;
    vocabulary: string[];
    samplesPerWord: number;

    // Word selection
    selectedWords: string[];
    isWordSelectionOpen: boolean;

    // Collection state
    samples: TrainingSample[];
    isRecording: boolean;
    recordingProgress: number; // 0-100

    // Countdown state
    isCountdown: boolean;
    countdownValue: number; // 5, 4, 3, 2, 1
    isPaused: boolean;

    // UI state
    showPreview: boolean;
    previewData: any[] | null;
    isWordComplete: boolean; // Persists until user clicks 'Start Next Word'

    // Actions
    setCurrentWord: (word: string, index: number) => void;
    setSelectedWords: (words: string[]) => void;
    setWordSelectionOpen: (open: boolean) => void;
    startCountdown: () => void;
    startRecording: () => void;
    stopRecording: () => void;
    setRecordingProgress: (progress: number) => void;
    setCountdownValue: (value: number) => void;
    setIsCountdown: (counting: boolean) => void;
    pauseTraining: () => void;
    resumeTraining: () => void;
    confirmSample: (frames: any[]) => void;
    retrySample: () => void;
    downloadCache: () => void;
    nextWord: () => void;
    resetWord: () => void;
    setWordComplete: (complete: boolean) => void;
}

export const useTrainingStore = create<TrainingState>((set, get) => ({
    // Initial state
    currentWord: FULL_VOCABULARY[0],
    currentWordIndex: 0,
    vocabulary: FULL_VOCABULARY,
    samplesPerWord: 40,
    selectedWords: FULL_VOCABULARY, // All words by default
    isWordSelectionOpen: false,
    samples: [],
    isRecording: false,
    recordingProgress: 0,
    isCountdown: false,
    countdownValue: 5,
    isPaused: false,
    showPreview: false,
    previewData: null,
    isWordComplete: false,

    // Actions
    setCurrentWord: (word, index) => set({
        currentWord: word,
        currentWordIndex: index
    }),

    setSelectedWords: (words) => set({
        selectedWords: words,
        vocabulary: words,
        currentWord: words[0],
        currentWordIndex: 0,
        samples: []
    }),

    setWordSelectionOpen: (open) => set({ isWordSelectionOpen: open }),

    setCountdownValue: (value) => set({ countdownValue: value }),

    setIsCountdown: (counting) => set({ isCountdown: counting }),

    startCountdown: () => set({
        isCountdown: true,
        countdownValue: 3,
        showPreview: false
    }),

    startRecording: () => set({
        isRecording: true,
        recordingProgress: 0,
        showPreview: false
    }),

    stopRecording: () => set({ isRecording: false }),

    setRecordingProgress: (progress) => set({ recordingProgress: progress }),

    pauseTraining: () => set({ isPaused: true }),

    resumeTraining: () => set({ isPaused: false }),

    confirmSample: (frames) => {
        const { samples, currentWord, samplesPerWord, nextWord } = get();
        const newSample: TrainingSample = {
            frames,
            timestamp: Date.now()
        };

        const updatedSamples = [...samples, newSample];
        set({
            samples: updatedSamples,
            showPreview: false,
            previewData: null
        });

        // Auto-advance check handled in UI components
    },

    retrySample: () => set({
        showPreview: false,
        previewData: null
    }),

    downloadCache: () => {
        const { currentWord, samples } = get();

        // Create JSON blob
        const data = {
            word: currentWord,
            samples: samples.map(s => s.frames),
            metadata: {
                sampleCount: samples.length,
                timestamp: Date.now(),
                version: '1.0'
            }
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], {
            type: 'application/json'
        });

        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${currentWord.toLowerCase()}_cache.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        // User must manually click "Start Next Word" to proceed
    },

    nextWord: () => {
        const { currentWordIndex, vocabulary } = get();
        const nextIndex = currentWordIndex + 1;

        if (nextIndex < vocabulary.length) {
            set({
                currentWordIndex: nextIndex,
                currentWord: vocabulary[nextIndex],
                samples: [],
                showPreview: false,
                previewData: null,
                isWordComplete: false // Reset for next word
            });
        } else {
            alert('All words completed! 🎉');
        }
    },

    resetWord: () => set({ samples: [], isWordComplete: false }),

    setWordComplete: (complete) => set({ isWordComplete: complete })
}));
