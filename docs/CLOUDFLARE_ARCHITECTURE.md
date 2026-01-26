# Cloudflare Architecture Explanation

## How It Works

### Cloudflare Pages = Pages + Workers Combined

When you deploy to **Cloudflare Pages**, you get:

1. **Static Site Hosting** (for Next.js frontend)
   - HTML, CSS, JavaScript files
   - Images and assets
   - Served from global CDN

2. **Pages Functions** (automatic Workers)
   - Next.js API routes (`/app/api/**/*.ts`) automatically become **Pages Functions**
   - Pages Functions run on the **Workers runtime**
   - No separate deployment needed!

### Architecture Diagram

```
User Request → Cloudflare Edge
              ↓
         ┌────────────────────────────┐
         │   Cloudflare Pages         │
         │                            │
         │  ┌──────────────────────┐  │
         │  │  Static Assets       │  │
         │  │  (HTML, CSS, JS)     │  │
         │  └──────────────────────┘  │
         │                            │
         │  ┌──────────────────────┐  │
         │  │  Pages Functions     │  │ ← These ARE Workers!
         │  │  (API Routes)        │  │
         │  │  - /api/predict      │  │
         │  │  - /api/transcribe   │  │
         │  │  - /api/translate    │  │
         │  │  - /api/room         │  │
         │  └──────────────────────┘  │
         └────────────────────────────┘
                      ↓
         ┌────────────────────────────┐
         │  Cloudflare Workers AI     │ ← Called via API
         │  - Whisper (ASR)           │
         │  - Llama-3 (NLP)           │
         └────────────────────────────┘
                      ↓
         ┌────────────────────────────┐
         │  Google Colab Backend      │
         │  (via Cloudflare Tunnel)   │
         │  - ML Model Inference      │
         └────────────────────────────┘
```

## What Gets Deployed Where

### Single `wrangler pages deploy` Command:

```bash
npm run pages:deploy
```

This deploys:
- ✅ Next.js frontend (static)
- ✅ API routes as Pages Functions (Workers)
- ✅ All configuration from wrangler.toml

### You DO NOT need separate deployments for:
- ❌ Cloudflare Workers (included in Pages)
- ❌ Workers AI (accessed via API with tokens)
- ❌ Cloudflare Tunnel (runs in Colab)

### You WOULD need separate Workers deployment for:
- **Durable Objects** (advanced WebSocket room state)
  - Currently using in-memory Map
  - Future enhancement: Deploy standalone Worker with Durable Objects
  - Would require: `wrangler deploy` (not `wrangler pages deploy`)

## Code Flow Example

### User makes sign → Translation

1. **Frontend** (Cloudflare Pages Static)
   ```
   Browser → MediaPipe extracts landmarks
   ```

2. **Pages Function = Worker** (automatic)
   ```typescript
   // /app/api/predict/route.ts runs as a Worker
   POST /api/predict
   → Receives landmarks
   → Calls Colab backend via fetch()
   → Returns prediction
   ```

3. **Workers AI** (called via API)
   ```typescript
   // /app/api/transcribe/route.ts
   POST /api/transcribe
   → Calls Cloudflare API: 
     fetch('https://api.cloudflare.com/.../whisper')
   → Returns transcription
   ```

## Deployment Types Comparison

| Type | What It Is | When Deployed | Example |
|------|-----------|---------------|---------|
| **Cloudflare Pages** | Static site + Functions | `npx wrangler pages deploy` | Our entire app |
| **Pages Functions** | Serverless functions (Workers) | Automatically with Pages | Our API routes |
| **Standalone Workers** | Separate Worker scripts | `npx wrangler deploy` | (Not used yet) |
| **Durable Objects** | Stateful Workers | Requires standalone Worker | (Future: rooms) |
| **Workers AI** | Cloudflare's AI models | Just API calls | Whisper, Llama |

## Why This Is Awesome

1. **Single Deployment**
   - One command deploys everything
   - Frontend + Backend together
   - No microservice complexity

2. **Zero Configuration**
   - API routes automatically become Workers
   - No separate routing setup
   - Works out of the box

3. **Free Tier**
   - 100,000 requests/day for Pages Functions
   - Same free tier as regular Workers
   - No upcharges

## Our Current Setup

### ✅ What We're Using:
- **Cloudflare Pages**: Full app deployment
- **Pages Functions**: API routes (automatic Workers)
- **Workers AI**: Called via REST API
- **No standalone Workers needed**

### 🔮 Future Enhancement (Optional):
If we wanted **persistent room state** across server restarts:

```typescript
// Would require standalone Worker deployment
// workers/rooms.ts
export class SignSpeakRoom implements DurableObject {
  // Persistent WebSocket connections
  // Survives server restarts
}
```

Then deploy with:
```bash
npx wrangler deploy workers/rooms.ts
```

But for now, **in-memory rooms work fine** for demos!

## Summary

**Question**: "Do we need to deploy Cloudflare Workers separately?"

**Answer**: **No!**
- Cloudflare Pages automatically converts API routes to Workers
- They're called "Pages Functions" but use the Workers runtime
- `npm run pages:deploy` handles everything
- Workers AI is accessed via API calls, not deployment

**One command does it all**: 
```bash
npm run pages:deploy
# Deploys: Frontend + API Routes (as Workers) + Configuration
```
