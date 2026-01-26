import Link from 'next/link';

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

        {/* Language Selection */}
        <div className="max-w-4xl mx-auto mb-12">
          <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-6 text-center">
            Select Sign Language
          </h2>
          <div className="grid md:grid-cols-2 gap-6">
            {/* ISL Card */}
            <Link href="/app?lang=isl">
              <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-300 cursor-pointer border-2 border-transparent hover:border-blue-500">
                <div className="text-5xl mb-4 text-center">🇮🇳</div>
                <h3 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-2 text-center">
                  Indian Sign Language
                </h3>
                <p className="text-gray-600 dark:text-gray-400 text-center">
                  ISL • 263 signs vocabulary
                </p>
              </div>
            </Link>

            {/* ASL Card */}
            <Link href="/app?lang=asl">
              <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-300 cursor-pointer border-2 border-transparent hover:border-indigo-500">
                <div className="text-5xl mb-4 text-center">🇺🇸</div>
                <h3 className="text-2xl font-bold text-gray-800 dark:text-gray-200 mb-2 text-center">
                  American Sign Language
                </h3>
                <p className="text-gray-600 dark:text-gray-400 text-center">
                  ASL • 2,000 signs vocabulary
                </p>
              </div>
            </Link>
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
    </main>
  );
}
