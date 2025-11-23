import { useState } from 'react'
import { AlertCircle, Settings } from 'lucide-react'
import { useProjectContext } from '../../contexts/ProjectContext'
import { useWorkspaceContext } from '../../contexts/WorkspaceContext'
import { Select } from '../ui/Select'
import { Alert, AlertDescription } from '../ui/Alert'
import { Skeleton } from '../ui/Skeleton'
import { Button } from '../ui/Button'
import { WorkspaceSetup } from '../workspace/WorkspaceSetup'

export function TopBar() {
  const { selectedProjectId, setSelectedProjectId, projects, isLoading, isError, error } =
    useProjectContext()
  const { workspacePath } = useWorkspaceContext()
  const [showWorkspaceSettings, setShowWorkspaceSettings] = useState(false)

  // Don't show error if workspace isn't configured - that's handled by WorkspaceProvider
  const shouldShowError = isError && error && !error.message?.includes('503')

  return (
    <>
      <header className="sticky top-0 z-10 flex h-16 items-center justify-between border-b border-border bg-card px-6">
        <div className="flex items-center gap-6">
          <h1 className="text-xl font-bold text-foreground">Flowlib</h1>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-muted-foreground">Current Project:</span>

          {shouldShowError ? (
            <Alert variant="destructive" className="w-64 py-2">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription className="text-xs">
                {error?.message || 'Failed to load projects'}
              </AlertDescription>
            </Alert>
          ) : isLoading ? (
            <Skeleton className="h-10 w-64" />
          ) : (
            <Select
              value={selectedProjectId}
              onChange={(e) => setSelectedProjectId(e.target.value)}
              disabled={projects.length === 0}
              className="w-64"
              aria-label="Select current project"
            >
              {projects.length === 0 ? (
                <option value="">No projects available</option>
              ) : (
                projects.map((project) => (
                  <option key={project.id} value={project.id}>
                    {project.name}
                  </option>
                ))
              )}
            </Select>
          )}

          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowWorkspaceSettings(true)}
            className="h-8 px-3"
            title={`Workspace: ${workspacePath || 'Not configured'}`}
          >
            <Settings className="h-4 w-4 mr-2" />
            <span className="hidden sm:inline">Workspace</span>
          </Button>
        </div>
      </header>

      {showWorkspaceSettings && (
        <WorkspaceSetup
          onComplete={() => {
            setShowWorkspaceSettings(false)
            // Page will reload after workspace is changed
          }}
          initialPath={workspacePath}
          isEditMode={true}
        />
      )}
    </>
  )
}
