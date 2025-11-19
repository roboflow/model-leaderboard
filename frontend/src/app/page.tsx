import type { Metadata } from 'next'
import ObjectDetectionClient from '@/components/ObjectDetectionClient'

export const metadata: Metadata = {
  title: 'Computer Vision Model Leaderboard | Object Detection Benchmarks',
  description: 'Compare object detection models like YOLO, RT-DETR, and D-FINE on COCO 2017 dataset. Filter by architecture, parameters, license, and performance metrics like mAP and F1 score.',
  robots: {
    index: false,
    follow: true,
  },
  openGraph: {
    title: 'Computer Vision Model Leaderboard | Object Detection Benchmarks',
    description: 'Compare object detection models like YOLO, RT-DETR, and D-FINE on COCO 2017 dataset. Filter by architecture, parameters, license, and performance metrics like mAP and F1 score.',
    url: 'https://roboflow.github.io/model-leaderboard/',
  },
  twitter: {
    title: 'Computer Vision Model Leaderboard | Object Detection Benchmarks',
    description: 'Compare object detection models like YOLO, RT-DETR, and D-FINE on COCO 2017 dataset. Filter by architecture, parameters, license, and performance metrics like mAP and F1 score.',
  },
}

export default function ObjectDetectionPage() {
  return <ObjectDetectionClient />
}