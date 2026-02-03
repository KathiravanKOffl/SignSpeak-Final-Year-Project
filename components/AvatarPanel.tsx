'use client';

export default function AvatarPanel() {
    return (
        <div className="flex flex-col h-full bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
            {/* Header */}
            <div className="px-6 py-4 border-b border-slate-100 bg-slate-50 flex justify-between items-center">
                <h2 className="font-semibold text-slate-700">Digital Avatar</h2>
                <div className="text-xs font-medium text-slate-400 uppercase tracking-wider">User 2</div>
            </div>

            {/* Content */}
            <div className="flex-1 bg-gradient-to-b from-slate-50 to-slate-100 flex flex-col items-center justify-center p-8 text-center relative overflow-hidden">

                {/* Simple CSS-only placeholder avatar */}
                <div className="relative w-48 h-48 mb-8">
                    <div className="absolute inset-0 bg-blue-100 rounded-full blur-3xl opacity-50 animate-pulse" />
                    <div className="relative w-full h-full bg-white border-4 border-white shadow-xl rounded-full overflow-hidden flex items-end justify-center">
                        {/* Head */}
                        <div className="w-20 h-24 bg-slate-200 rounded-full mb-2 absolute top-8" />
                        {/* Body */}
                        <div className="w-36 h-20 bg-slate-300 rounded-t-full absolute bottom-0" />
                    </div>
                </div>

                <h3 className="text-lg font-semibold text-slate-700 mb-2">Speak to Sign</h3>
                <p className="text-sm text-slate-500 max-w-[200px] leading-relaxed">
                    Avatar will animate when User 2 speaks words into the system.
                </p>

                <div className="mt-8 px-4 py-2 bg-white/50 backdrop-blur border border-slate-200 rounded-full text-xs text-slate-400 font-mono">
                    STATUS: IDLE
                </div>

            </div>
        </div>
    );
}
