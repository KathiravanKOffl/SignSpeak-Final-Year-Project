import { NextRequest, NextResponse } from 'next/server';

interface TranslateRequest {
    text: string;
    source_language: string; // 'en', 'hi', 'ta'
    target_language: 'isl' | 'asl';
}

export async function POST(request: NextRequest) {
    try {
        const body: TranslateRequest = await request.json();

        if (!body.text) {
            return NextResponse.json(
                { error: 'Missing text to translate' },
                { status: 400 }
            );
        }

        // Use Cloudflare Workers AI for text-to-gloss translation
        const CLOUDFLARE_ACCOUNT_ID = process.env.CLOUDFLARE_ACCOUNT_ID;
        const CLOUDFLARE_API_TOKEN = process.env.CLOUDFLARE_API_TOKEN;

        if (!CLOUDFLARE_ACCOUNT_ID || !CLOUDFLARE_API_TOKEN) {
            return NextResponse.json(
                { error: 'Cloudflare credentials not configured' },
                { status: 503 }
            );
        }

        // Create prompt for LLM to convert text to sign gloss
        const prompt = `Convert the following ${body.source_language.toUpperCase()} text to ${body.target_language.toUpperCase()} sign language gloss notation.

Text: "${body.text}"

Rules:
1. Use uppercase for sign glosses
2. Remove articles (a, an, the)
3. Use present tense verbs
4. Maintain semantic meaning
5. Follow ${body.target_language.toUpperCase()} grammar structure

Gloss:`;

        // Call Cloudflare Workers AI
        const response = await fetch(
            `https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/meta/llama-3.1-8b-instruct`,
            {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${CLOUDFLARE_API_TOKEN}`,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: prompt,
                    max_tokens: 200,
                }),
            }
        );

        if (!response.ok) {
            throw new Error(`Cloudflare AI error: ${response.statusText}`);
        }

        const result = await response.json();
        const gloss = result.result?.response || '';

        // Clean up the gloss output
        const cleanGloss = gloss
            .split('\n')[0] // Take first line
            .trim()
            .replace(/[^\w\s-]/g, '') // Remove special chars except hyphens
            .toUpperCase();

        return NextResponse.json({
            gloss: cleanGloss,
            original_text: body.text,
            target_language: body.target_language,
            source_language: body.source_language,
        });

    } catch (error) {
        console.error('Translation error:', error);
        return NextResponse.json(
            {
                error: 'Translation failed',
                details: error instanceof Error ? error.message : 'Unknown error',
            },
            { status: 500 }
        );
    }
}
