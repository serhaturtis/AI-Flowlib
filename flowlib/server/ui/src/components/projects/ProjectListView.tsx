import { useState } from 'react'
import { CheckCircle2, Settings, ExternalLink, Trash2 } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Button } from '../ui/Button'
import { Spinner } from '../ui/Spinner'
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

export interface ProjectListViewProps {
  projects: ProjectMetadata[]
  isValidating: boolean
  validatingProjectId: string | undefined
  onValidate: (projectId: string) => void
  onDelete: (projectId: string) => void
  deletingProjectId?: string
}

/**
 * Table/list view layout for projects.
 */
export function ProjectListView({
  projects,
  isValidating,
  validatingProjectId,
  onValidate,
  onDelete,
  deletingProjectId,
}: ProjectListViewProps) {
  const [deleteDialogProjectId, setDeleteDialogProjectId] = useState<string | null>(null)
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr className="border-b border-border">
            <th className="text-left p-3 font-medium">Name</th>
            <th className="text-left p-3 font-medium">Description</th>
            <th className="text-left p-3 font-medium">Path</th>
            <th className="text-left p-3 font-medium">Updated</th>
            <th className="text-left p-3 font-medium">Actions</th>
          </tr>
        </thead>
        <tbody>
          {projects.map((project) => (
            <tr key={project.id} className="border-b border-border hover:bg-muted/50">
              <td className="p-3 font-medium">{project.name}</td>
              <td className="p-3 text-muted-foreground">{project.description}</td>
              <td className="p-3">
                <code className="text-sm bg-muted px-2 py-1 rounded">{project.path}</code>
              </td>
              <td className="p-3 text-sm text-muted-foreground">
                {new Date(project.updated_at).toLocaleString(undefined, {
                  dateStyle: 'medium',
                  timeStyle: 'short',
                })}
              </td>
              <td className="p-3">
                <div className="flex gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => onValidate(project.id)}
                    disabled={isValidating && validatingProjectId === project.id}
                  >
                    {isValidating && validatingProjectId === project.id ? (
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
                  <AlertDialog
                    open={deleteDialogProjectId === project.id}
                    onOpenChange={(open: boolean) => setDeleteDialogProjectId(open ? project.id : null)}
                  >
                    <AlertDialogTrigger asChild>
                      <Button
                        type="button"
                        variant="destructive"
                        size="sm"
                        disabled={deletingProjectId === project.id}
                      >
                        {deletingProjectId === project.id ? (
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
                          Are you sure you want to delete "{project.name}"? This will permanently
                          delete the project directory and all its contents. This action cannot be
                          undone.
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                          onClick={() => {
                            onDelete(project.id)
                            setDeleteDialogProjectId(null)
                          }}
                          className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                        >
                          Delete
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
