import path from "path";
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Skip type checking and linting during build
  typescript: {
    ignoreBuildErrors: true,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },

  // Webpack configuration - CRITICAL for module resolution
  webpack: (config) => {
    // Ensure @ alias resolves correctly in all environments
    config.resolve.alias = {
      ...config.resolve.alias,
      '@': path.resolve(__dirname, './'),
    };

    return config;
  },

  // Important: Output must be compatible with Cloudflare Pages
  // @cloudflare/next-on-pages will convert this
  images: {
    unoptimized: true, // Cloudflare Images optimization will handle this
  },
};

export default nextConfig;
