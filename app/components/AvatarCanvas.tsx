'use client';

import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, ContactShadows } from '@react-three/drei';
import Avatar from './Avatar';
import { Suspense } from 'react';

export default function AvatarCanvas() {
    return (
        <div className="w-full h-[500px] border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-xl overflow-hidden bg-gray-50 dark:bg-gray-900 relative">
            <div className="absolute top-4 left-4 z-10 bg-black/50 text-white px-3 py-1 rounded text-sm">
                Avatar Preview
            </div>

            <Canvas camera={{ position: [0, 0, 5], fov: 40 }} shadows>
                <Suspense fallback={null}>
                    <Environment preset="city" />
                    <ambientLight intensity={0.5} />
                    <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} shadow-mapSize={2048} castShadow />

                    <Avatar />

                    <ContactShadows resolution={1024} scale={10} blur={1} opacity={0.5} far={10} color="#000000" />
                    <OrbitControls enablePan={false} minPolarAngle={Math.PI / 2.5} maxPolarAngle={Math.PI / 2} />
                </Suspense>
            </Canvas>
        </div>
    );
}
