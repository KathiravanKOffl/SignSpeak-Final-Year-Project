'use client';

import { useSearchParams } from 'next/navigation';
import { Suspense } from 'react';

function OutputContent() {
    const searchParams = useSearchParams();
    const roomId = searchParams.get('room') || '';

    return (
        <div className="min-h-screen bg-gradient-to-br from-indigo-900 to-purple-900 text-white">
            {/* Minimal Header */}
            <header className="absolute top-0 left-0 right-0 z-10 p-6">
                <div className="flex items-center justify-between">
                    <div className="bg-black/30 backdrop-blur px-4 py-2 rounded-lg">
                        <span className="text-sm text-gray-300">Room:</span>
                        <span className="ml-2 font-mono text-green-400">{roomId}</span>
                    </div>
                </div>
            </header>

            {/* Full Screen Avatar Display */}
            <main className="h-screen flex flex-col items-center justify-center p-8">
                {/* Avatar Area */}
                <div className="flex-1 w-full flex items-center justify-center">
                    <div className="text-center">
                        <div className="text-9xl mb-8">🧑</div>
                        <p className="text-3xl font-light">3D Avatar Display</p>
                        <p className="text-xl text-gray-300 mt-4">Sign animations will render here</p>
                    </div>
                </div>

                {/* Large Text Display */}
                <div className="w-full max-w-4xl bg-black/30 backdrop-blur rounded-2xl p-8 mb-8">
                    <div className="text-center">
                        <p className="text-4xl font-light text-gray-300">
                            Translated text appears here...
                        </p>
                    </div>
                </div>

                {/* Status Indicator */}
                <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
                    <span className="text-sm text-gray-300">Receiving from control panel</span>
                </div>
            </main>
        </div>
    );
}

export default function OutputPage() {
    return (
        <Suspense fallback={<div>Loading...</div>}>
            <OutputContent />
        </Suspense>
    );
}
