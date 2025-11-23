import { createContext, useContext, ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getWorkspacePath } from '../services/workspace'
import { WorkspaceSetup } from '../components/workspace/WorkspaceSetup'
import { AxiosError } from 'axios'

interface WorkspaceContextValue {
  isConfigured: boolean
  isLoading: boolean
  workspacePath: string | null
}

const WorkspaceContext = createContext<WorkspaceContextValue | undefined>(undefined)

export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const {
    data: workspaceData,
    isLoading,
    isError,
    error,
  } = useQuery({
    queryKey: ['workspace', 'path'],
    queryFn: getWorkspacePath,
    retry: false,
    refetchOnWindowFocus: false,
  })

  const isConfigured = !!workspaceData && !isError
  const workspacePath = workspaceData?.path || null

  // Check if error is a 503 (workspace not configured)
  const isWorkspaceNotConfigured =
    isError &&
    error instanceof AxiosError &&
    error.response?.status === 503

  // Show setup dialog if workspace is not configured (503 error)
  if (isWorkspaceNotConfigured) {
    return (
      <WorkspaceSetup
        onComplete={() => {
          // Query will be refetched after page reload
        }}
      />
    )
  }

  // Still loading workspace status - don't render children yet
  if (isLoading) {
    return null
  }

  // Workspace is configured or no error (shouldn't happen but handle gracefully)
  return (
    <WorkspaceContext.Provider value={{ isConfigured, isLoading, workspacePath }}>
      {children}
    </WorkspaceContext.Provider>
  )
}

export function useWorkspaceContext() {
  const context = useContext(WorkspaceContext)
  if (context === undefined) {
    throw new Error('useWorkspaceContext must be used within WorkspaceProvider')
  }
  return context
}

