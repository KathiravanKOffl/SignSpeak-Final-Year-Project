'use client';

import { useRef, useState, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Mesh, Color } from 'three';

export type Emotion = 'neutral' | 'happy' | 'sad' | 'angry' | 'surprised' | 'confused';

export interface AvatarProps {
    emotion?: Emotion;
    speaking?: boolean;
    currentGloss?: string;
}

// Emotion to color mapping
const EMOTION_COLORS: Record<Emotion, string> = {
    neutral: '#808080',
    happy: '#22c55e',
    sad: '#3b82f6',
    angry: '#ef4444',
    surprised: '#eab308',
    confused: '#a855f7'
};

export default function Avatar({ emotion = 'neutral', speaking = false, currentGloss }: AvatarProps) {
    const meshRef = useRef<Mesh>(null);
    const headRef = useRef<Mesh>(null);
    const [hovered, setHover] = useState(false);

    // Compute color based on emotion
    const bodyColor = useMemo(() => new Color(EMOTION_COLORS[emotion]), [emotion]);
    const headColor = useMemo(() => {
        const baseColor = new Color('navajowhite');
        // Slightly tint head based on emotion
        return speaking ? baseColor.lerp(bodyColor, 0.2) : baseColor;
    }, [emotion, speaking, bodyColor]);

    useFrame((state) => {
        if (meshRef.current) {
            // Gentle idle animation
            meshRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;

            // Breathing animation
            const breathe = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.02;
            meshRef.current.scale.x = breathe;
            meshRef.current.scale.z = breathe;
        }

        if (headRef.current) {
            // Head bob when speaking
            if (speaking) {
                headRef.current.position.y = 1.5 + Math.sin(state.clock.elapsedTime * 8) * 0.02;
            } else {
                headRef.current.position.y = 1.5;
            }
        }
    });

    return (
        <group position={[0, -1, 0]}>
            {/* Body */}
            <mesh
                ref={meshRef}
                onPointerOver={() => setHover(true)}
                onPointerOut={() => setHover(false)}>
                <boxGeometry args={[1, 2, 0.5]} />
                <meshStandardMaterial color={hovered ? 'hotpink' : bodyColor} />
            </mesh>

            {/* Head */}
            <mesh ref={headRef} position={[0, 1.5, 0]}>
                <sphereGeometry args={[0.6, 32, 32]} />
                <meshStandardMaterial color={headColor} />
            </mesh>

            {/* Current gloss display would go here */}
        </group>
    );
}
