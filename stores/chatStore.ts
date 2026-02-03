import { create } from 'zustand';

export type UserRole = 'signer' | 'speaker';

export interface Message {
    id: string;
    sender: UserRole;
    text: string;
    timestamp: number;
}

interface ChatState {
    messages: Message[];
    isRecording: boolean; // For speaker mic
    addMessage: (sender: UserRole, text: string) => void;
    setRecording: (isRecording: boolean) => void;
    clearChat: () => void;
}

export const useChatStore = create<ChatState>((set) => ({
    messages: [],
    isRecording: false,

    addMessage: (sender, text) => set((state) => ({
        messages: [
            ...state.messages,
            {
                id: crypto.randomUUID(),
                sender,
                text,
                timestamp: Date.now(),
            }
        ]
    })),

    setRecording: (isRecording) => set({ isRecording }),
    clearChat: () => set({ messages: [] })
}));
