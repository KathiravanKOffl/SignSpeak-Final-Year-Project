import { create } from 'zustand';

// Tier 1 vocabulary (start with these)
export const TIER1_VOCABULARY = [
    'I', 'you', 'he', 'she',
    'want', 'need', 'go', 'come',
    'to', 'can'
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

    // Collection state
    samples: TrainingSample[];
    isRecording: boolean;
    recordingProgress: number; // 0-100

    // UI state
    showPreview: boolean;
    previewData: any[] | null;

    // Actions
    setCurrentWord: (word: string, index: number) => void;
    startRecording: () => void;
    stopRecording: () => void;
    setRecordingProgress: (progress: number) => void;
    confirmSample: (frames: any[]) => void;
    retrySample: () => void;
    downloadCache: () => void;
    nextWord: () => void;
    resetWord: () => void;
}

export const useTrainingStore = create<TrainingState>((set, get) => ({
    // Initial state
    currentWord: TIER1_VOCABULARY[0],
    currentWordIndex: 0,
    vocabulary: TIER1_VOCABULARY,
    samplesPerWord: 40,
    samples: [],
    isRecording: false,
    recordingProgress: 0,
    showPreview: false,
    previewData: null,

    // Actions
    setCurrentWord: (word, index) => set({
        currentWord: word,
        currentWordIndex: index
    }),

    startRecording: () => set({
        isRecording: true,
        recordingProgress: 0,
        showPreview: false
    }),

    stopRecording: () => set({ isRecording: false }),

    setRecordingProgress: (progress) => set({ recordingProgress: progress }),

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

        // Auto-advance if we've reached target
        if (updatedSamples.length >= samplesPerWord) {
            // Trigger download
            setTimeout(() => {
                get().downloadCache();
            }, 500);
        }
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

        // Move to next word
        setTimeout(() => {
            get().nextWord();
        }, 1000);
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
                previewData: null
            });
        } else {
            alert('All words completed! 🎉');
        }
    },

    resetWord: () => set({ samples: [] })
}));
