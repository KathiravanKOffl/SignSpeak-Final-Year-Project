'use client';

import { useChatStore, UserRole } from '@/stores/chatStore';
import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function ChatPanel() {
    const { messages, addMessage } = useChatStore();
    const bottomRef = useRef<HTMLDivElement>(null);
    const [inputText, setInputText] = useState('');

    // Auto-scroll to bottom
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSend = (e?: React.FormEvent) => {
        if (e) e.preventDefault();
        if (!inputText.trim()) return;

        // Simulate User 2 (Speaker) sending a message
        addMessage('speaker', inputText);
        setInputText('');
    };

    return (
        <div className="flex flex-col h-full bg-slate-50/50 backdrop-blur-sm rounded-2xl border border-slate-200 overflow-hidden relative">

            {/* Header */}
            <div className="px-6 py-4 bg-white border-b border-slate-100 shadow-sm z-10">
                <h2 className="font-semibold text-slate-800">Conversation</h2>
                <p className="text-xs text-slate-400 mt-0.5">Live Translation</p>
            </div>

            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
                {messages.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center text-slate-300">
                        <div className="w-16 h-16 bg-slate-100 rounded-2xl flex items-center justify-center mb-4">
                            <svg className="w-8 h-8 text-slate-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                            </svg>
                        </div>
                        <p className="text-sm font-medium">No messages yet</p>
                        <p className="text-xs mt-1">Start signing or typing...</p>
                    </div>
                ) : (
                    <AnimatePresence initial={false}>
                        {messages.map((msg) => (
                            <motion.div
                                key={msg.id}
                                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                                animate={{ opacity: 1, y: 0, scale: 1 }}
                                className={`flex ${msg.sender === 'speaker' ? 'justify-end' : 'justify-start'}`}
                            >
                                <div className={`max-w-[80%] ${msg.sender === 'speaker' ? 'items-end' : 'items-start'} flex flex-col`}>
                                    <span className="text-[10px] uppercase font-bold text-slate-400 mb-1 px-1">
                                        {msg.sender === 'signer' ? 'Signer (User 1)' : 'Speaker (User 2)'}
                                    </span>
                                    <div className={`px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-sm ${msg.sender === 'speaker'
                                            ? 'bg-blue-600 text-white rounded-br-none'
                                            : 'bg-white border border-slate-200 text-slate-700 rounded-bl-none'
                                        }`}>
                                        {msg.text}
                                    </div>
                                    <span className="text-[9px] text-slate-300 mt-1 px-1">
                                        {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                    </span>
                                </div>
                            </motion.div>
                        ))}
                    </AnimatePresence>
                )}
                <div ref={bottomRef} />
            </div>

            {/* Input Area (Mocking User 2 Mic/Text) */}
            <div className="p-4 bg-white border-t border-slate-100">
                <form onSubmit={handleSend} className="relative flex items-center gap-3">
                    <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none">
                        <svg className="w-5 h-5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                        </svg>
                    </div>
                    <input
                        type="text"
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        placeholder="Type to speak as User 2..."
                        className="w-full pl-10 pr-12 py-3 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all placeholder:text-slate-400 text-slate-700"
                    />
                    <button
                        type="submit"
                        disabled={!inputText.trim()}
                        className="absolute right-2 p-1.5 bg-blue-600 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors"
                    >
                        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                        </svg>
                    </button>
                </form>
            </div>

        </div>
    );
}
