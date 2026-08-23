import { Analytics } from '@vercel/analytics/next'
import type { Metadata, Viewport } from 'next'
import Providers from './Providers'
import Sidebar from '../components/common/Sidebar'
import './globals.css'

export const metadata: Metadata = {
  title: 'Relay Desk — Support workspace',
  description: 'A modern AI workspace for thoughtful, fast customer support.',
  generator: 'v0.app',
  icons: {
    icon: [
      {
        url: '/icon-light-32x32.png',
        media: '(prefers-color-scheme: light)',
      },
      {
        url: '/icon-dark-32x32.png',
        media: '(prefers-color-scheme: dark)',
      },
      {
        url: '/icon.svg',
        type: 'image/svg+xml',
      },
    ],
    apple: '/apple-icon.png',
  },
}

export const viewport: Viewport = {
  colorScheme: 'light dark',
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#f7f6f0' },
    { media: '(prefers-color-scheme: dark)', color: '#20252d' },
  ],
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="h-dvh min-h-dvh w-screen overflow-hidden antialiased" suppressHydrationWarning>
        <Providers>
          <div className="flex h-full w-full aurora-shell">
            <Sidebar />
            {children}
          </div>
          {process.env.NODE_ENV === 'production' && <Analytics />}
        </Providers>
      </body>
    </html>
  )
}
