import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'edge';

interface CreateRoomRequest {
    language?: 'isl' | 'asl';
}

// In-memory room storage (in production, use Durable Objects)
const rooms = new Map<string, {
    id: string;
    created_at: number;
    language: string;
    devices: string[];
}>();

export async function POST(request: NextRequest) {
    try {
        const body: CreateRoomRequest = await request.json();

        // Generate unique room ID using Web Crypto API
        const array = new Uint8Array(6);
        crypto.getRandomValues(array);
        const roomId = Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');

        // Create room
        const room = {
            id: roomId,
            created_at: Date.now(),
            language: body.language || 'isl',
            devices: [],
        };

        rooms.set(roomId, room);

        // Auto-cleanup after 24 hours
        setTimeout(() => {
            rooms.delete(roomId);
        }, 24 * 60 * 60 * 1000);

        return NextResponse.json({
            room_id: roomId,
            language: room.language,
            urls: {
                input: `/input?room=${roomId}`,
                control: `/control?room=${roomId}`,
                output: `/output?room=${roomId}`,
            },
            created_at: room.created_at,
        });

    } catch (error) {
        console.error('Room creation error:', error);
        return NextResponse.json(
            {
                error: 'Failed to create room',
                details: error instanceof Error ? error.message : 'Unknown error',
            },
            { status: 500 }
        );
    }
}

export async function GET(request: NextRequest) {
    try {
        const searchParams = request.nextUrl.searchParams;
        const roomId = searchParams.get('room');

        if (!roomId) {
            return NextResponse.json(
                { error: 'Missing room ID' },
                { status: 400 }
            );
        }

        const room = rooms.get(roomId);

        if (!room) {
            return NextResponse.json(
                { error: 'Room not found' },
                { status: 404 }
            );
        }

        return NextResponse.json({
            id: room.id,
            language: room.language,
            created_at: room.created_at,
            devices: room.devices,
            age_minutes: Math.floor((Date.now() - room.created_at) / 60000),
        });

    } catch (error) {
        console.error('Room info error:', error);
        return NextResponse.json(
            {
                error: 'Failed to get room info',
                details: error instanceof Error ? error.message : 'Unknown error',
            },
            { status: 500 }
        );
    }
}
