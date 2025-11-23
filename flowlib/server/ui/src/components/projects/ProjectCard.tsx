import { useState } from 'react'
import { CheckCircle2, Settings, ExternalLink, Trash2 } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../ui/Card'
import { Button } from '../ui/Button'
import { Spinner } from '../ui/Spinner'
import { Stack } from '../layout/Stack'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '../ui/AlertDialog'
import type { ProjectMetadata } from '../../services/projects'

export interface ProjectCardProps {
  project: ProjectMetadata
  isValidating: boolean
  onValidate: (projectId: string) => void
  onDelete: (projectId: string) => void
  isDeleting?: boolean
}

/**
 * Reusable project card component for grid view.
 */
export function ProjectCard({ project, isValidating, onValidate, onDelete, isDeleting }: ProjectCardProps) {
  const [isDialogOpen, setIsDialogOpen] = useState(false)

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <CardTitle className="text-lg truncate">{project.name}</CardTitle>
            <CardDescription className="mt-1 line-clamp-2">
              {project.description || 'No description'}
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Stack spacing="sm">
          <div className="text-xs text-muted-foreground">
            <code className="bg-muted px-2 py-1 rounded text-xs break-all">{project.path}</code>
          </div>
          <div className="text-xs text-muted-foreground">
            Updated:{' '}
            {new Date(project.updated_at).toLocaleDateString(undefined, {
              dateStyle: 'medium',
            })}
          </div>
          <div className="flex flex-wrap gap-2 pt-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => onValidate(project.id)}
              disabled={isValidating}
              className="flex-1"
            >
              {isValidating ? (
                <>
                  <Spinner size="sm" className="mr-2" />
                  Validating…
                </>
              ) : (
                <>
                  <CheckCircle2 className="h-3 w-3 mr-1" />
                  Validate
                </>
              )}
            </Button>
            <Button type="button" variant="outline" size="sm" asChild>
              <Link to={`/configs?project=${project.id}`}>
                <Settings className="h-3 w-3 mr-1" />
                Configs
              </Link>
            </Button>
            <Button type="button" variant="outline" size="sm" asChild>
              <Link to={`/agents?project=${project.id}`}>
                <ExternalLink className="h-3 w-3 mr-1" />
                Agents
              </Link>
            </Button>
            <AlertDialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
              <AlertDialogTrigger asChild>
                <Button type="button" variant="destructive" size="sm" disabled={isDeleting}>
                  {isDeleting ? (
                    <>
                      <Spinner size="sm" className="mr-2" />
                      Deleting…
                    </>
                  ) : (
                    <>
                      <Trash2 className="h-3 w-3 mr-1" />
                      Delete
                    </>
                  )}
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Delete Project</AlertDialogTitle>
                  <AlertDialogDescription>
                    Are you sure you want to delete "{project.name}"? This will permanently delete
                    the project directory and all its contents. This action cannot be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    onClick={() => {
                      onDelete(project.id)
                      setIsDialogOpen(false)
                    }}
                    className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                  >
                    Delete
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        </Stack>
      </CardContent>
    </Card>
  )
}
