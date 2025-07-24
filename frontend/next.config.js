/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true
  },
  basePath: '/model-leaderboard',
  assetPrefix: '/model-leaderboard',
}

module.exports = nextConfig