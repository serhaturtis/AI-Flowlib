import { createContext, useContext, useMemo, type ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchProjects, type ProjectMetadata } from '../services/projects'
import { useProjectSelection } from '../hooks/useProjectSelection'
import { useWorkspaceContext } from './WorkspaceContext'

interface ProjectContextValue {
  selectedProjectId: string
  setSelectedProjectId: (projectId: string) => void
  selectedProject: ProjectMetadata | undefined
  projects: ProjectMetadata[]
  isLoading: boolean
  isError: boolean
  error: Error | null
}

const ProjectContext = createContext<ProjectContextValue | null>(null)

export function ProjectProvider({ children }: { children: ReactNode }) {
  // Check workspace status first before fetching projects
  // Note: This component only renders when workspace is configured (WorkspaceProvider handles that)
  const { isConfigured } = useWorkspaceContext()

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['projects'],
    queryFn: fetchProjects,
    enabled: isConfigured, // Only fetch if workspace is configured
    retry: false,
  })

  const projects = useMemo(() => data?.projects || [], [data?.projects])

  const projectOptions = useMemo(
    () => projects.map((p) => ({ id: p.id, name: p.name })),
    [projects]
  )

  const [selectedProjectId, setSelectedProjectId] = useProjectSelection(projectOptions)

  const selectedProject = useMemo(
    () => projects.find((p) => p.id === selectedProjectId),
    [projects, selectedProjectId]
  )

  const value = useMemo(
    () => ({
      selectedProjectId,
      setSelectedProjectId,
      selectedProject,
      projects,
      isLoading,
      isError,
      error: error as Error | null,
    }),
    [selectedProjectId, setSelectedProjectId, selectedProject, projects, isLoading, isError, error]
  )

  return <ProjectContext.Provider value={value}>{children}</ProjectContext.Provider>
}

// eslint-disable-next-line react-refresh/only-export-components
export function useProjectContext() {
  const context = useContext(ProjectContext)
  if (!context) {
    throw new Error('useProjectContext must be used within a ProjectProvider')
  }
  return context
}
