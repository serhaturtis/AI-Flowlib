import { type ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { LayoutDashboard, FolderOpen, Settings, Bot, Brain, Terminal } from 'lucide-react'
import { cn } from '../../utils/cn'

interface NavItem {
  path: string
  label: string
  icon: ReactNode
}

const navItems: NavItem[] = [
  { path: '/', label: 'Dashboard', icon: <LayoutDashboard className="h-5 w-5" /> },
  { path: '/projects', label: 'Projects', icon: <FolderOpen className="h-5 w-5" /> },
  { path: '/configs', label: 'Configs', icon: <Settings className="h-5 w-5" /> },
  { path: '/agents', label: 'Agents', icon: <Bot className="h-5 w-5" /> },
  { path: '/knowledge', label: 'Knowledge Plugins', icon: <Brain className="h-5 w-5" /> },
  { path: '/run-console', label: 'Run Console', icon: <Terminal className="h-5 w-5" /> },
]

export function Sidebar() {
  const location = useLocation()

  return (
    <aside className="flex h-full w-64 flex-col border-r border-border bg-card">
      <nav className="flex-1 space-y-1 p-4">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path
          return (
            <Link
              key={item.path}
              to={item.path}
              className={cn(
                'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground',
              )}
            >
              {item.icon}
              <span>{item.label}</span>
            </Link>
          )
        })}
      </nav>
    </aside>
  )
}

