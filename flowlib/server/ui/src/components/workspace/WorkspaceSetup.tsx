import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { FolderOpen } from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '../ui/Dialog'
import { Button } from '../ui/Button'
import { Input } from '../ui/Input'
import { Label } from '../ui/Label'
import { Alert, AlertDescription } from '../ui/Alert'
import { AlertCircle } from 'lucide-react'
import { Spinner } from '../ui/Spinner'
import { setWorkspacePath } from '../../services/workspace'
import { DirectoryBrowser } from './DirectoryBrowser'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/Tabs'

interface WorkspaceSetupProps {
  onComplete: () => void
  initialPath?: string | null
  isEditMode?: boolean
}

export function WorkspaceSetup({ onComplete, initialPath, isEditMode = false }: WorkspaceSetupProps) {
  const [workspacePath, setWorkspacePathValue] = useState(initialPath || '')
  const [error, setError] = useState<string | null>(null)
  const [tab, setTab] = useState<'manual' | 'browse'>(isEditMode ? 'manual' : 'browse')
  const [cancelled, setCancelled] = useState(false)

  const { mutate: configureWorkspace, isPending } = useMutation({
    mutationFn: (path: string) => setWorkspacePath(path),
    onSuccess: () => {
      // Workspace configured successfully
      onComplete()
      // Reload page to reinitialize services (only if not cancelled)
      if (!cancelled) {
        window.location.reload()
      }
    },
    onError: (err: unknown) => {
      const message =
        err instanceof Error
          ? err.message
          : 'Failed to configure workspace. Please try again.'
      setError(message)
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)

    if (!workspacePath.trim()) {
      setError('Workspace path is required')
      return
    }

    configureWorkspace(workspacePath.trim())
  }

  const handleDirectorySelect = (path: string) => {
    setWorkspacePathValue(path)
    setTab('manual') // Switch to manual tab to show the selected path
  }

  const defaultPath = `${window.location.hostname === 'localhost' ? '~' : '/var'}/flowlib-workspace`

  return (
    <Dialog open={true}>
      <DialogContent className="sm:max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>{isEditMode ? 'Change Workspace Path' : 'Workspace Configuration'}</DialogTitle>
          <DialogDescription>
            {isEditMode
              ? 'Update where your Flowlib projects are stored. The server will need to be restarted for changes to take effect.'
              : 'Configure where your Flowlib projects will be stored. This can be changed later via settings.'}
          </DialogDescription>
        </DialogHeader>

        <Tabs value={tab} onValueChange={(v) => setTab(v as 'manual' | 'browse')} className="flex-1 flex flex-col min-h-0">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="browse">
              <FolderOpen className="h-4 w-4 mr-2" />
              Browse
            </TabsTrigger>
            <TabsTrigger value="manual">Manual Entry</TabsTrigger>
          </TabsList>

          <TabsContent value="browse" className="flex-1 min-h-0 mt-4">
            <DirectoryBrowser
              onSelect={handleDirectorySelect}
              initialPath={isEditMode && workspacePath ? workspacePath : null}
            />
          </TabsContent>

          <TabsContent value="manual" className="mt-4">
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="workspace-path">Workspace Path</Label>
                <Input
                  id="workspace-path"
                  type="text"
                  value={workspacePath}
                  onChange={(e) => setWorkspacePathValue(e.target.value)}
                  placeholder={defaultPath}
                  disabled={isPending}
                  autoFocus
                />
                <p className="text-sm text-muted-foreground">
                  Enter an absolute path or relative path (relative to server directory). The directory
                  will be created if it doesn't exist.
                </p>
              </div>

              {error && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              <div className="flex justify-end gap-2">
                {isEditMode && (
                  <Button
                    type="button"
                    variant="outline"
                    onClick={(e) => {
                      e.preventDefault()
                      setCancelled(true)
                      onComplete()
                    }}
                    disabled={isPending}
                  >
                    Cancel
                  </Button>
                )}
                <Button type="submit" disabled={isPending || !workspacePath.trim()}>
                  {isPending ? (
                    <>
                      <Spinner className="mr-2 h-4 w-4" />
                      {isEditMode ? 'Updating...' : 'Configuring...'}
                    </>
                  ) : (
                    isEditMode ? 'Update Workspace' : 'Configure Workspace'
                  )}
                </Button>
              </div>
            </form>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  )
}

