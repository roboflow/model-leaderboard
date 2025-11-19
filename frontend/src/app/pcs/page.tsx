import type { Metadata } from 'next'
import PCSClient from '@/components/PCSClient'

export const metadata: Metadata = {
  title: 'Computer Vision Model Leaderboard | Promptable Concept Segmentation (PCS) Benchmarks',
  description: 'Compare Promptable Concept Segmentation (PCS) models that you can prompt with text, points, or image regions.',
  robots: {
    index: false,
    follow: true,
  },
  openGraph: {
    title: 'Computer Vision Model Leaderboard | Promptable Concept Segmentation (PCS) Benchmarks',
    description: 'Compare Promptable Concept Segmentation (PCS) models that you can prompt with text, points, or image regions.',
    url: 'https://roboflow.github.io/model-leaderboard/pcs',
  },
  twitter: {
    title: 'Computer Vision Model Leaderboard | Promptable Concept Segmentation (PCS) Benchmarks',
    description: 'Compare Promptable Concept Segmentation (PCS) models that you can prompt with text, points, or image regions.',
  },
}

export default function PCSPage() {
  return <PCSClient />
}