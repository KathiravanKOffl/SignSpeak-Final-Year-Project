import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'edge';

interface TranscribeRequest {
    audio: string; // Base64 encoded audio
    language?: string; // Target language for transcription
}

export async function POST(request: NextRequest) {
    try {
        const body: TranscribeRequest = await request.json();

        if (!body.audio) {
            return NextResponse.json(
                { error: 'Missing audio data' },
                { status: 400 }
            );
        }

        // Use Cloudflare Workers AI for speech recognition
        const CLOUDFLARE_ACCOUNT_ID = process.env.CLOUDFLARE_ACCOUNT_ID;
        const CLOUDFLARE_API_TOKEN = process.env.CLOUDFLARE_API_TOKEN;

        if (!CLOUDFLARE_ACCOUNT_ID || !CLOUDFLARE_API_TOKEN) {
            return NextResponse.json(
                { error: 'Cloudflare credentials not configured' },
                { status: 503 }
            );
        }

        // Decode base64 audio
        const audioBuffer = Buffer.from(body.audio, 'base64');

        // Call Cloudflare Workers AI Whisper model
        const response = await fetch(
            `https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/openai/whisper`,
            {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${CLOUDFLARE_API_TOKEN}`,
                    'Content-Type': 'application/octet-stream',
                },
                body: audioBuffer,
            }
        );

        if (!response.ok) {
            throw new Error(`Cloudflare AI error: ${response.statusText}`);
        }

        const result = await response.json();

        return NextResponse.json({
            text: result.result?.text || '',
            language: result.result?.language || 'en',
            confidence: 1.0, // Whisper doesn't provide confidence scores
        });

    } catch (error) {
        console.error('Transcription error:', error);
        return NextResponse.json(
            {
                error: 'Transcription failed',
                details: error instanceof Error ? error.message : 'Unknown error',
            },
            { status: 500 }
        );
    }
}
