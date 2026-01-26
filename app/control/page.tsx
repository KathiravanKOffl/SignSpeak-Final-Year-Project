'use client';

import { useSearchParams } from 'next/navigation';
import { Suspense } from 'react';

function ControlContent() {
    const searchParams = useSearchParams();
    const roomId = searchParams.get('room') || '';

    return (
        <div className="min-h-screen bg-gray-900 text-white">
            {/* Header */}
            <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold">Control Panel</h1>
                        <p className="text-sm text-gray-400">Room: <span className="font-mono text-green-400">{roomId}</span></p>
                    </div>
                    <button className="px-4 py-2 bg-red-600 hover:bg-red-500 rounded-lg transition">
                        End Session
                    </button>
                </div>
            </header>

            {/* Main Dashboard */}
            <main className="p-6">
                <div className="grid lg:grid-cols-3 gap-6">
                    {/* Connected Devices */}
                    <div className="lg:col-span-1 bg-gray-800 rounded-xl p-6">
                        <h2 className="text-lg font-semibold mb-4">Connected Devices</h2>
                        <div className="space-y-3">
                            <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                                <div>
                                    <div className="font-medium">📸 Input Device</div>
                                    <div className="text-xs text-gray-400">Streaming landmarks</div>
                                </div>
                                <div className="w-2 h-2 bg-green-500 rounded-full" />
                            </div>
                            <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                                <div>
                                    <div className="font-medium">🎮 Control (You)</div>
                                    <div className="text-xs text-gray-400">Orchestrating</div>
                                </div>
                                <div className="w-2 h-2 bg-green-500 rounded-full" />
                            </div>
                            <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                                <div>
                                    <div className="font-medium">🖥️ Output Display</div>
                                    <div className="text-xs text-gray-400">Rendering avatar</div>
                                </div>
                                <div className="w-2 h-2 bg-green-500 rounded-full" />
                            </div>
                        </div>
                    </div>

                    {/* Transcript + Settings */}
                    <div className="lg:col-span-2 space-y-6">
                        {/* Cloud Status */}
                        <div className="bg-gray-800 rounded-xl p-6">
                            <h2 className="text-lg font-semibold mb-4">Cloud Services Status</h2>
                            <div className="grid md:grid-cols-2 gap-4">
                                <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                                    <span>Colab Backend</span>
                                    <span className="text-green-400">✓ Online</span>
                                </div>
                                <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                                    <span>Cloudflare Workers</span>
                                    <span className="text-green-400">✓ Online</span>
                                </div>
                            </div>
                        </div>

                        {/* Transcript */}
                        <div className="bg-gray-800 rounded-xl p-6">
                            <h2 className="text-lg font-semibold mb-4">Conversation Transcript</h2>
                            <div className="space-y-2 h-64 overflow-y-auto">
                                <div className="text-gray-400 text-center py-8">
                                    Translations will appear here...
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}

export default function ControlPage() {
    return (
        <Suspense fallback={<div>Loading...</div>}>
            <ControlContent />
        </Suspense>
    );
}
