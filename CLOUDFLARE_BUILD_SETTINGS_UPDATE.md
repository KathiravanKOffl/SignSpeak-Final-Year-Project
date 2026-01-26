# 🚨 IMPORTANT: Update Cloudflare Pages Build Settings

After the restructure, you **MUST** update your build settings in the Cloudflare Pages dashboard.

## Steps to Update Build Settings:

### 1. Go to Cloudflare Pages Dashboard
- Navigate to: https://dash.cloudflare.com/
- Click: **Workers & Pages**
- Select your project: **signspeak**

### 2. Update Build Configuration
- Click the **Settings** tab
- Scroll down to **Build configuration**
- Click **Edit configuration** button

### 3. Change These Settings:

| Setting | OLD Value ❌ | NEW Value ✅ |
|---------|-------------|-------------|
| **Build command** | `cd signspeak-web && npm install && npm run pages:build` | `npm install && npm run build` |
| **Build output directory** | `signspeak-web/.vercel/output/static` | `.next` |
| **Root directory** | `/` or `signspeak-web` | `/` (leave blank) |

### 4. Save and Redeploy
- Click **Save** button
- Go to **Deployments** tab
- Click **Retry deployment** on the latest failed build

---

## Why This Is Needed:

Cloudflare Pages saves your build settings in its database. Even though we updated the code and `wrangler.toml`, the dashboard settings override everything.

The error shows it's still trying to run:
```
cd signspeak-web && npm install && npm run pages:build
```

But `signspeak-web/` directory no longer exists!

---

## Alternative: Delete and Recreate Project

If updating settings doesn't work, you can:

1. **Delete the signspeak project** in Cloudflare Pages
2. **Create a new project** with GitHub integration
3. Use the NEW build settings from the start:
   - Build command: `npm install && npm run build`
   - Output: `.next`
   - Root: `/`

---

## Verify After Update:

Once you save the new settings, the next deployment should show:
```
✅ Executing user command: npm install && npm run build
✅ Installing dependencies from package.json (React 18)
✅ Building Next.js app
✅ Success!
```

Let me know once you've updated the settings and I'll help monitor the deployment! 🚀
