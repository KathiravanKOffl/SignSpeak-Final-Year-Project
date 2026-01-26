'use client';

import { useSearchParams } from 'next/navigation';
import { Suspense } from 'react';

function InputContent() {
    const searchParams = useSearchParams();
    const roomId = searchParams.get('room') || '';

    return (
        <div className="min-h-screen bg-black text-white">
            {/* Minimal Header */}
            <header className="absolute top-0 left-0 right-0 z-10 p-4">
                <div className="flex items-center justify-between">
                    <div className="bg-black/50 backdrop-blur px-4 py-2 rounded-lg">
                        <span className="text-sm text-gray-300">Room:</span>
                        <span className="ml-2 font-mono text-green-400">{roomId}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                        <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
                        <span className="text-sm text-gray-300">Connected</span>
                    </div>
                </div>
            </header>

            {/* Full Screen Camera */}
            <main className="h-screen flex items-center justify-center">
                <div className="w-full h-full relative">
                    <div className="absolute inset-0 flex items-center justify-center text-gray-500">
                        <div className="text-center">
                            <div className="text-9xl mb-6">📸</div>
                            <p className="text-2xl">Input Device</p>
                            <p className="text-lg mt-2 text-gray-400">Camera feed with skeleton overlay</p>
                        </div>
                    </div>
                </div>
            </main>

            {/* Status Footer */}
            <footer className="absolute bottom-0 left-0 right-0 p-4">
                <div className="bg-black/50 backdrop-blur rounded-lg p-3 text-center">
                    <p className="text-sm text-gray-300">
                        Streaming landmarks to control panel
                    </p>
                </div>
            </footer>
        </div>
    );
}

export default function InputPage() {
    return (
        <Suspense fallback={<div>Loading...</div>}>
            <InputContent />
        </Suspense>
    );
}
