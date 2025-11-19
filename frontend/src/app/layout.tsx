import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { GoogleAnalytics } from '@next/third-parties/google'
import { ThemeProvider } from "@/components/theme-provider"
import { Footer } from "@/components/Footer"
import { Header } from "@/components/Header"

const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  preload: true,
  variable: '--font-inter'
})

export const metadata: Metadata = {
  title: 'Computer Vision Model Leaderboard',
  description: 'Benchmarks showing the performance of popular computer vision models across metrics like mAP and F1 score.',
  icons: {
    icon: [
      { url: './favicon.ico', sizes: 'any' },
      { url: './favicon.svg', type: 'image/svg+xml' },
      { url: './favicon.png', type: 'image/png', sizes: '32x32' },
    ],
    apple: [{ url: './apple-touch-icon.png', sizes: '180x180' }],
  },
  openGraph: {
    title: 'Computer Vision Model Leaderboard',
    description: 'Benchmarks showing the performance of popular computer vision models across metrics like mAP and F1 score.',
    url: 'https://roboflow.github.io/model-leaderboard/',
    siteName: 'Computer Vision Model Leaderboard',
    images: [
      {
        url: '/static/roboflow-model-leaderboard-og.webp',
        width: 1200,
        height: 630,
        alt: 'Computer Vision Model Leaderboard - Performance benchmarks for YOLO, RT-DETR and other models',
      },
    ],
    locale: 'en_US',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Computer Vision Model Leaderboard',
    description: 'Benchmarks showing the performance of popular computer vision models across metrics like mAP and F1 score.',
    images: ['/static/roboflow-model-leaderboard-og.webp'],
    creator: '@roboflow',
    site: '@roboflow',
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning className="h-full">
      <head>
        <link rel="icon" href="./favicon.ico" sizes="any" />
        <link rel="icon" href="./favicon.svg" type="image/svg+xml" />
        <link rel="icon" href="./favicon.png" type="image/png" />
        <link rel="apple-touch-icon" href="./apple-touch-icon.png" />
      </head>
      <body className={`flex flex-col min-h-full ${inter.className}`}>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <div className="orb-container">
            <div className="orb orb-1"></div>
            {/* <div className="orb orb-2"></div> */}
            {/* <div className="orb orb-3"></div> */}
          </div>

          <Header />
          <main className="flex-1 flex flex-col gap-4 sm:gap-10">
            {children}
          </main>
          <Footer />
        </ThemeProvider>

        <GoogleAnalytics gaId="G-KH26XBHQR9" />
      </body>
    </html>
  )
}
