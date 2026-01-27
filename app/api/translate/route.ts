import { NextRequest, NextResponse } from 'next/server';
import { convertToGloss } from '../../lib/textToGloss';
import { detectEmotion, getEmotionConfidence } from '../../lib/sentimentAnalysis';

export const runtime = 'edge';

interface TranslateRequest {
    text: string;
    source_language?: string; // 'en', 'hi', 'ta'
    target_language?: 'isl' | 'asl';
    includeEmotion?: boolean;
    useLLM?: boolean;
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

        const text = body.text.trim();
        const targetLang = body.target_language || 'asl';

        // Always compute local rule-based translation first
        const localResult = convertToGloss(text);
        const emotionResult = body.includeEmotion !== false ? getEmotionConfidence(text) : null;

        // Use Cloudflare Workers AI for enhanced translation if requested
        const CLOUDFLARE_ACCOUNT_ID = process.env.CLOUDFLARE_ACCOUNT_ID;
        const CLOUDFLARE_API_TOKEN = process.env.CLOUDFLARE_API_TOKEN;

        // If no Cloudflare credentials or LLM not requested, use local translation
        if (!CLOUDFLARE_ACCOUNT_ID || !CLOUDFLARE_API_TOKEN || !body.useLLM) {
            return NextResponse.json({
                gloss: localResult.gloss.join(' '),
                glossArray: localResult.gloss,
                isQuestion: localResult.isQuestion,
                questionType: localResult.questionType,
                emotion: emotionResult?.emotion || 'neutral',
                emotionConfidence: emotionResult?.confidence || 0,
                original_text: body.text,
                target_language: targetLang,
                source_language: body.source_language || 'en',
                method: 'rule-based'
            });
        }

        // Create prompt for LLM to convert text to sign gloss
        const prompt = `Convert the following ${(body.source_language || 'EN').toUpperCase()} text to ${targetLang.toUpperCase()} sign language gloss notation.

Text: "${body.text}"

Rules:
1. Use uppercase for sign glosses
2. Remove articles (a, an, the)
3. Use present tense verbs
4. Maintain semantic meaning
5. Follow ${targetLang.toUpperCase()} grammar structure

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
            // Fallback to local translation if LLM fails
            return NextResponse.json({
                gloss: localResult.gloss.join(' '),
                glossArray: localResult.gloss,
                isQuestion: localResult.isQuestion,
                questionType: localResult.questionType,
                emotion: emotionResult?.emotion || 'neutral',
                emotionConfidence: emotionResult?.confidence || 0,
                original_text: body.text,
                target_language: targetLang,
                source_language: body.source_language || 'en',
                method: 'rule-based-fallback'
            });
        }

        const result = await response.json();
        const gloss = result.result?.response || '';

        // Clean up the gloss output
        const cleanGloss = gloss
            .split('\n')[0] // Take first line
            .trim()
            .replace(/[^\w\s-]/g, '') // Remove special chars except hyphens
            .toUpperCase();

        const glossArray = cleanGloss.split(/\s+/).filter((w: string) => w.length > 0);

        return NextResponse.json({
            gloss: cleanGloss,
            glossArray: glossArray,
            isQuestion: localResult.isQuestion,
            questionType: localResult.questionType,
            emotion: emotionResult?.emotion || 'neutral',
            emotionConfidence: emotionResult?.confidence || 0,
            original_text: body.text,
            target_language: targetLang,
            source_language: body.source_language || 'en',
            method: 'llm-enhanced'
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

// Health check
export async function GET() {
    return NextResponse.json({
        status: 'ok',
        service: 'text-to-gloss',
        methods: ['rule-based', 'llm-enhanced'],
        languages: ['asl', 'isl']
    });
}
