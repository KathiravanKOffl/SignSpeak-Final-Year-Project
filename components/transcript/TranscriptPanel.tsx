'use client';

import { TranscriptMessage } from '@/stores/appStore';

interface TranscriptPanelProps {
    messages: TranscriptMessage[];
    onClear?: () => void;
}

export function TranscriptPanel({ messages, onClear }: TranscriptPanelProps) {
    return (
        <div className="bg-gray-800 border-t border-gray-700 p-4 h-48">
            <div className="flex items-center justify-between mb-3">
                <h2 className="font-semibold text-white">Conversation Transcript</h2>
                <button
                    onClick={onClear}
                    className="text-sm text-gray-400 hover:text-white transition"
                >
                    Clear
                </button>
            </div>
            <div className="space-y-2 overflow-y-auto h-32">
                {messages.length === 0 ? (
                    <div className="text-gray-400 text-sm text-center py-4">
                        Translations will appear here...
                    </div>
                ) : (
                    messages.map((message) => (
                        <div
                            key={message.id}
                            className="bg-gray-700 rounded-lg p-3 text-sm"
                        >
                            <div className="flex items-center justify-between mb-1">
                                <span className="text-xs text-gray-400">
                                    {message.type === 'sign' ? '🤟 Sign' : message.type === 'speech' ? '🗣️ Speech' : '📢 System'}
                                </span>
                                <span className="text-xs text-gray-500">
                                    {new Date(message.timestamp).toLocaleTimeString()}
                                </span>
                            </div>
                            <p className="text-white">{message.text}</p>
                            {message.confidence && (
                                <div className="mt-1 text-xs text-gray-400">
                                    Confidence: {(message.confidence * 100).toFixed(0)}%
                                </div>
                            )}
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}
