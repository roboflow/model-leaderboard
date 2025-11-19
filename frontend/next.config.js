/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true
  },
  basePath: '/model-leaderboard-fork',
  assetPrefix: '/model-leaderboard-fork',
}

module.exports = nextConfig