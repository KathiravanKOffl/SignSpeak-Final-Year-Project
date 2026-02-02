'use client';

import Link from 'next/link';
import AvatarPlaceholder from './components/AvatarPlaceholder';

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-indigo-950">
      <div className="container mx-auto px-4 py-16">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-600 dark:from-blue-400 dark:to-indigo-400 mb-4">
            SignSpeak
          </h1>
          <p className="text-xl text-gray-700 dark:text-gray-300 max-w-2xl mx-auto">
            Bidirectional Real-Time Sign Language Translation
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
            Breaking communication barriers with AI-powered translation
          </p>
        </div>

        {/* Avatar Preview */}
        <div className="max-w-2xl mx-auto mb-16">
          <AvatarPlaceholder emotion="neutral" />
        </div>

        {/* Language Selection */}
        <div className="max-w-4xl mx-auto mb-12">
          <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-6 text-center">
            Select Sign Language
          </h2>
          <div className="grid md:grid-cols-2 gap-6">
            {/* ISL Card - Active ✅ */}
            <Link href="/app?lang=isl">
              <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-300 cursor-pointer border-2 border-indigo-500 ring-2 ring-indigo-500/30 relative">
                <div className="absolute top-3 right-3 bg-green-500 text-white text-xs font-bold px-2 py-1 rounded-full flex items-center gap-1">
                  <span className="w-1.5 h-1.5 bg-white rounded-full animate-pulse" />
                  Active
                </div>
                <div className="text-5xl mb-4 text-center">🇮🇳</div>
                <h3 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-2 text-center">
                  Indian Sign Language
                </h3>
                <p className="text-gray-600 dark:text-gray-400 text-center">
                  ISL • 123 signs • 76% accuracy
                </p>
                <div className="mt-4 flex items-center justify-center gap-2 text-xs text-indigo-600 dark:text-indigo-400">
                  <span>🎯</span>
                  <span>Production ready</span>
                </div>
              </div>
            </Link>

            {/* ASL Card - Coming Soon */}
            <div className="bg-white/50 dark:bg-gray-800/50 rounded-2xl p-8 shadow-lg relative overflow-hidden cursor-not-allowed opacity-75">
              <div className="absolute top-3 right-3 bg-amber-500 text-white text-xs font-bold px-2 py-1 rounded-full">
                Coming Soon
              </div>
              <div className="text-5xl mb-4 text-center grayscale">🇺🇸</div>
              <h3 className="text-2xl font-bold text-gray-600 dark:text-gray-400 mb-2 text-center">
                American Sign Language
              </h3>
              <p className="text-gray-500 dark:text-gray-500 text-center">
                ASL • Future Release
              </p>
            </div>
          </div>

        </div>

        {/* Features Grid */}
        <div className="max-w-6xl mx-auto grid md:grid-cols-3 gap-6 mb-12">
          <div className="bg-white/50 dark:bg-gray-800/50 backdrop-blur rounded-xl p-6">
            <div className="text-3xl mb-3">⚡</div>
            <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">
              Real-Time Translation
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              &lt;1.2s end-to-end latency for seamless communication
            </p>
          </div>

          <div className="bg-white/50 dark:bg-gray-800/50 backdrop-blur rounded-xl p-6">
            <div className="text-3xl mb-3">🔒</div>
            <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">
              Privacy-Preserving
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Edge processing - only landmarks transmitted, not video
            </p>
          </div>

          <div className="bg-white/50 dark:bg-gray-800/50 backdrop-blur rounded-xl p-6">
            <div className="text-3xl mb-3">📱</div>
            <h3 className="font-semibold text-gray-800 dark:text-gray-200 mb-2">
              Multi-Device Mode
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Distribute across devices with shareable room links
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-sm text-gray-500 dark:text-gray-500">
          <p>Built with zero-cost infrastructure • Cloudflare Pages + Workers AI</p>
          <p className="mt-2">Supporting accessibility and breaking communication barriers</p>
        </div>
      </div>
    </main >
  );
}
