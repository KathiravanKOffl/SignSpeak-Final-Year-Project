'use client';

import { useState } from 'react';
import { Settings } from 'lucide-react';
import CameraPanel from '@/components/CameraPanel';
import ChatPanel from '@/components/ChatPanel';
import AvatarPanel from '@/components/AvatarPanel';
import SettingsModal from '@/components/SettingsModal';
import { useInference } from '@/hooks/useInference';

export default function Home() {
  const [showSettings, setShowSettings] = useState(false);
  const { testConnection } = useInference();

  const handleSaveSettings = (url: string) => {
    localStorage.setItem('backend_url', url);
    // Refresh page to reinitialize with new URL
    window.location.reload();
  };

  const handleTestConnection = async (url: string): Promise<boolean> => {
    // Temporarily store URL for testing
    const oldUrl = localStorage.getItem('backend_url');
    localStorage.setItem('backend_url', url);

    const result = await testConnection();

    // Restore old URL if test failed
    if (!result && oldUrl) {
      localStorage.setItem('backend_url', oldUrl);
    }

    return result;
  };

  return (
    <main className="min-h-screen bg-[#F8F9FA] text-slate-900 overflow-hidden flex flex-col font-sans">

      {/* Top Navigation / Brand */}
      <header className="h-16 bg-white border-b border-slate-200 flex items-center px-8 justify-between shadow-sm z-20">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center text-white font-bold text-lg">
            S
          </div>
          <div>
            <h1 className="text-lg font-bold text-slate-800 tracking-tight">SignSpeak</h1>
            <p className="text-[10px] text-slate-500 font-medium uppercase tracking-wider">Accessibility Suite</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="px-3 py-1 bg-green-50 text-green-700 rounded-full text-xs font-semibold border border-green-100 flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
            SYSTEM ONLINE
          </div>
          <button
            onClick={() => setShowSettings(true)}
            className="w-8 h-8 rounded-full bg-slate-100 hover:bg-slate-200 border border-slate-200 flex items-center justify-center transition-colors"
            title="Settings"
          >
            <Settings className="w-4 h-4 text-slate-600" />
          </button>
        </div>
      </header>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        onSave={handleSaveSettings}
        onTest={handleTestConnection}
      />

      {/* Main Content Grid */}
      <div className="flex-1 p-6 h-[calc(100vh-64px)]">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full max-w-[1920px] mx-auto">

          {/* Left: Signer Input */}
          <div className="h-full">
            <CameraPanel />
          </div>

          {/* Center: Conversation */}
          <div className="h-full">
            <ChatPanel />
          </div>

          {/* Right: Avatar */}
          <div className="h-full">
            <AvatarPanel />
          </div>

        </div>
      </div>

    </main>
  );
}
