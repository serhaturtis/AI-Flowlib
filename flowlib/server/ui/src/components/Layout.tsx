import { type ReactNode } from 'react'
import { Sidebar } from './layout/Sidebar'
import { TopBar } from './layout/TopBar'
import { Container } from './layout/Container'

interface LayoutProps {
  children: ReactNode
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <TopBar />
      <div className="flex flex-1">
        <Sidebar />
        <main className="flex-1 overflow-y-auto">
          <Container className="py-8">{children}</Container>
        </main>
      </div>
    </div>
  )
}
