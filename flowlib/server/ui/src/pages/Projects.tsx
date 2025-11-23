import { useMemo, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { FolderOpen, AlertCircle, Grid3x3, List, Plus } from 'lucide-react'
import { fetchProjects, deleteProject, ProjectMetadata } from '../services/projects'
import { Button } from '../components/ui/Button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/Card'
import { Alert, AlertDescription, AlertTitle } from '../components/ui/Alert'
import { Stack } from '../components/layout/Stack'
import { Skeleton } from '../components/ui/Skeleton'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '../components/ui/Dialog'
import { useProjectValidation } from '../hooks/projects/useProjectValidation'
import { ProjectGridView } from '../components/projects/ProjectGridView'
import { ProjectListView } from '../components/projects/ProjectListView'
import { ProjectCreationWizard } from '../components/projects/ProjectCreationWizard'
import { ValidationResults } from '../components/projects/ValidationResults'

export default function Projects() {
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [createDialogOpen, setCreateDialogOpen] = useState(false)
  const [deleteError, setDeleteError] = useState<string | null>(null)

  const queryClient = useQueryClient()

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['projects'],
    queryFn: fetchProjects,
  })

  const validationHook = useProjectValidation()

  const {
    mutate: triggerDelete,
    variables: deletingProjectId,
  } = useMutation({
    mutationFn: (projectId: string) => deleteProject(projectId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] })
      setDeleteError(null)
    },
    onError: (deleteErr: unknown) => {
      const message = deleteErr instanceof Error ? deleteErr.message : 'Failed to delete project.'
      setDeleteError(message)
    },
  })

  const handleDeleteProject = (projectId: string) => {
    setDeleteError(null)
    triggerDelete(projectId)
  }

  const handleWizardComplete = () => {
    setCreateDialogOpen(false)
  }

  const handleWizardCancel = () => {
    setCreateDialogOpen(false)
  }

  const sortedProjects = useMemo<ProjectMetadata[]>(() => {
    const projects = data?.projects ?? []
    return [...projects].sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
  }, [data?.projects])

  return (
    <Stack spacing="xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold tracking-tight">Projects</h1>
          <p className="text-muted-foreground mt-2">Manage your Flowlib projects and configurations</p>
        </div>
        <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Create Project
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-5xl max-h-[90vh] overflow-y-auto w-[90vw]">
            <DialogHeader>
              <DialogTitle>Create New Project</DialogTitle>
              <DialogDescription>
                Set up a new Flowlib project with guided assistance or start from scratch
              </DialogDescription>
            </DialogHeader>
            <ProjectCreationWizard
              onComplete={handleWizardComplete}
              onCancel={handleWizardCancel}
            />
          </DialogContent>
        </Dialog>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <FolderOpen className="h-5 w-5" />
                Project List
              </CardTitle>
              <CardDescription>
                {isLoading ? 'Loading projects…' : `Managing ${data?.total ?? 0} projects.`}
              </CardDescription>
            </div>
            {!isLoading && !isError && sortedProjects.length > 0 && (
              <div className="flex items-center gap-2">
                <Button
                  type="button"
                  variant={viewMode === 'grid' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setViewMode('grid')}
                  title="Grid view"
                >
                  <Grid3x3 className="h-4 w-4" />
                </Button>
                <Button
                  type="button"
                  variant={viewMode === 'list' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setViewMode('list')}
                  title="List view"
                >
                  <List className="h-4 w-4" />
                </Button>
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-2">
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-12 w-full" />
            </div>
          ) : isError ? (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{(error as Error).message}</AlertDescription>
            </Alert>
          ) : sortedProjects.length === 0 ? (
            <div className="text-center py-8">
              <FolderOpen className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">No projects found yet.</p>
            </div>
          ) : viewMode === 'grid' ? (
            <ProjectGridView
              projects={sortedProjects}
              isValidating={validationHook.isValidating}
              validatingProjectId={validationHook.validatingProjectId}
              onValidate={validationHook.handleValidateProject}
              onDelete={handleDeleteProject}
              deletingProjectId={deletingProjectId}
            />
          ) : (
            <ProjectListView
              projects={sortedProjects}
              isValidating={validationHook.isValidating}
              validatingProjectId={validationHook.validatingProjectId}
              onValidate={validationHook.handleValidateProject}
              onDelete={handleDeleteProject}
              deletingProjectId={deletingProjectId}
            />
          )}
        </CardContent>
      </Card>

      {validationHook.validationResult && <ValidationResults result={validationHook.validationResult} />}

      {validationHook.validationError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Validation Error</AlertTitle>
          <AlertDescription>{validationHook.validationError}</AlertDescription>
        </Alert>
      )}

      {deleteError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Delete Error</AlertTitle>
          <AlertDescription>{deleteError}</AlertDescription>
        </Alert>
      )}
    </Stack>
  )
}
