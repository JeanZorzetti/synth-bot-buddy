import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Abutre Dashboard - Delayed Martingale Bot',
  description: 'Real-time trading dashboard for Abutre bot (+40.25% ROI validated)',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  )
}
