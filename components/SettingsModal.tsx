'use client';

import { useState } from 'react';
import { X, Settings, Wifi, WifiOff } from 'lucide-react';

interface SettingsModalProps {
    isOpen: boolean;
    onClose: () => void;
    onSave: (url: string) => void;
    onTest: (url: string) => Promise<boolean>;
}

export default function SettingsModal({ isOpen, onClose, onSave, onTest }: SettingsModalProps) {
    const [url, setUrl] = useState(() => {
        if (typeof window !== 'undefined') {
            return localStorage.getItem('backend_url') || '';
        }
        return '';
    });
    const [testing, setTesting] = useState(false);
    const [testResult, setTestResult] = useState<'success' | 'error' | null>(null);
    const [errorMessage, setErrorMessage] = useState('');

    if (!isOpen) return null;

    const handleTest = async () => {
        if (!url.trim()) {
            setTestResult('error');
            setErrorMessage('Please enter a URL');
            return;
        }

        setTesting(true);
        setTestResult(null);
        setErrorMessage('');

        try {
            const success = await onTest(url);
            setTestResult(success ? 'success' : 'error');
            if (!success) {
                setErrorMessage('Connection failed. Check URL and try again.');
            }
        } catch (err) {
            setTestResult('error');
            setErrorMessage(err instanceof Error ? err.message : 'Connection failed');
        } finally {
            setTesting(false);
        }
    };

    const handleSave = () => {
        if (!url.trim()) {
            setErrorMessage('Please enter a URL');
            return;
        }
        onSave(url);
        onClose();
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md mx-4 overflow-hidden">
                {/* Header */}
                <div className="bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <Settings className="w-6 h-6 text-white" />
                        <h2 className="text-xl font-bold text-white">Backend Settings</h2>
                    </div>
                    <button
                        onClick={onClose}
                        className="text-white/80 hover:text-white transition-colors"
                    >
                        <X className="w-6 h-6" />
                    </button>
                </div>

                {/* Body */}
                <div className="p-6 space-y-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Backend URL
                        </label>
                        <input
                            type="url"
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            placeholder="https://your-tunnel.trycloudflare.com"
                            className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                        />
                        <p className="mt-2 text-xs text-gray-500">
                            Paste the cloudflared tunnel URL from your Kaggle inference notebook
                        </p>
                    </div>

                    {/* Test Result */}
                    {testResult && (
                        <div className={`p-3 rounded-lg flex items-center gap-2 ${testResult === 'success'
                                ? 'bg-green-50 text-green-700 border border-green-200'
                                : 'bg-red-50 text-red-700 border border-red-200'
                            }`}>
                            {testResult === 'success' ? (
                                <>
                                    <Wifi className="w-5 h-5" />
                                    <span className="text-sm font-medium">Connected successfully!</span>
                                </>
                            ) : (
                                <>
                                    <WifiOff className="w-5 h-5" />
                                    <span className="text-sm font-medium">
                                        {errorMessage || 'Connection failed'}
                                    </span>
                                </>
                            )}
                        </div>
                    )}

                    {/* Actions */}
                    <div className="flex gap-3 pt-2">
                        <button
                            onClick={handleTest}
                            disabled={testing || !url.trim()}
                            className="flex-1 px-4 py-3 bg-blue-100 text-blue-700 rounded-lg font-medium hover:bg-blue-200 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                        >
                            {testing ? 'Testing...' : 'Test Connection'}
                        </button>
                        <button
                            onClick={handleSave}
                            disabled={!url.trim() || testResult !== 'success'}
                            className="flex-1 px-4 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-medium hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                        >
                            Save & Close
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
