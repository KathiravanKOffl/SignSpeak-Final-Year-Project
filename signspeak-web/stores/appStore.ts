import { create } from 'zustand';
import { LandmarksData } from '@/hooks/useMediaPipe';

interface AppState {
    // Mode
    mode: 'unified' | 'multidevice';
    setMode: (mode: 'unified' | 'multidevice') => void;

    // Language
    language: 'isl' | 'asl';
    setLanguage: (language: 'isl' | 'asl') => void;

    // Room (for multi-device mode)
    roomId: string | null;
    setRoomId: (roomId: string | null) => void;

    // Landmarks
    currentLandmarks: LandmarksData | null;
    setCurrentLandmarks: (landmarks: LandmarksData | null) => void;

    // Transcript
    transcript: TranscriptMessage[];
    addTranscriptMessage: (message: TranscriptMessage) => void;
    clearTranscript: () => void;

    // Connection status
    isConnected: boolean;
    setIsConnected: (connected: boolean) => void;
}

export interface TranscriptMessage {
    id: string;
    timestamp: number;
    type: 'sign-to-text' | 'speech-to-sign';
    originalText: string;
    translatedText: string;
    confidence?: number;
}

export const useAppStore = create<AppState>((set) => ({
    // Mode
    mode: 'unified',
    setMode: (mode) => set({ mode }),

    // Language
    language: 'isl',
    setLanguage: (language) => set({ language }),

    // Room
    roomId: null,
    setRoomId: (roomId) => set({ roomId }),

    // Landmarks
    currentLandmarks: null,
    setCurrentLandmarks: (landmarks) => set({ currentLandmarks: landmarks }),

    // Transcript
    transcript: [],
    addTranscriptMessage: (message) =>
        set((state) => ({
            transcript: [...state.transcript, message],
        })),
    clearTranscript: () => set({ transcript: [] }),

    // Connection
    isConnected: false,
    setIsConnected: (connected) => set({ isConnected: connected }),
}));
