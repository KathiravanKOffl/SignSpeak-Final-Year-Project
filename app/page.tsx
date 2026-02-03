'use client';

import CameraPanel from '@/components/CameraPanel';
import ChatPanel from '@/components/ChatPanel';
import AvatarPanel from '@/components/AvatarPanel';

export default function Home() {
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
          <div className="w-8 h-8 rounded-full bg-slate-100 border border-slate-200" />
        </div>
      </header>

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
