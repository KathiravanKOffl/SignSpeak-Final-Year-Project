import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'edge';

interface LandmarksData {
    pose: number[][];
    leftHand: number[][];
    rightHand: number[][];
    face: number[][];
    confidence: number;
    timestamp: number;
}

interface PredictRequest {
    landmarks: LandmarksData;
    language: 'isl' | 'asl';
    top_k?: number;
}

export async function POST(request: NextRequest) {
    try {
        const body: PredictRequest = await request.json();

        // Validate request
        if (!body.landmarks) {
            return NextResponse.json(
                { error: 'Missing landmarks data' },
                { status: 400 }
            );
        }

        // Get Colab tunnel URL from environment
        const COLAB_TUNNEL_URL = process.env.COLAB_TUNNEL_URL;

        if (!COLAB_TUNNEL_URL) {
            return NextResponse.json(
                { error: 'Backend server URL not configured' },
                { status: 503 }
            );
        }

        // Call backend inference API - send full landmarks object
        const baseUrl = COLAB_TUNNEL_URL.replace(/\/+$/, ''); // Remove trailing slashes
        const response = await fetch(`${baseUrl}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                landmarks: body.landmarks,  // Send full object, not flattened
                language: body.language || 'asl',
                top_k: body.top_k || 5,
            }),
        });

        if (!response.ok) {
            throw new Error(`Backend error: ${response.statusText}`);
        }

        const prediction = await response.json();

        return NextResponse.json(prediction);

    } catch (error) {
        console.error('Prediction error:', error);
        return NextResponse.json(
            {
                error: 'Prediction failed',
                details: error instanceof Error ? error.message : 'Unknown error'
            },
            { status: 500 }
        );
    }
}

// Health check
export async function GET() {
    const COLAB_TUNNEL_URL = process.env.COLAB_TUNNEL_URL;

    if (!COLAB_TUNNEL_URL) {
        return NextResponse.json({
            status: 'error',
            message: 'Backend URL not configured',
        });
    }

    try {
        const baseUrl = COLAB_TUNNEL_URL.replace(/\/+$/, '');
        const response = await fetch(`${baseUrl}/health`, {
            method: 'GET',
        });

        const health = await response.json();

        return NextResponse.json({
            status: 'ok',
            backend: health,
            tunnel_url: COLAB_TUNNEL_URL,
        });
    } catch (error) {
        return NextResponse.json({
            status: 'error',
            message: 'Cannot reach backend',
            error: error instanceof Error ? error.message : 'Unknown error',
        });
    }
}
