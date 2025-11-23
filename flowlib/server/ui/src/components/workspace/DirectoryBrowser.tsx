import { useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useMutation } from '@tanstack/react-query'
import { FolderOpen, Folder, File, ChevronRight, ChevronUp, Home, Eye, EyeOff, FolderPlus } from 'lucide-react'
import { Button } from '../ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card'
import { Skeleton } from '../ui/Skeleton'
import { Alert, AlertDescription } from '../ui/Alert'
import { AlertCircle } from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '../ui/Dialog'
import { Input } from '../ui/Input'
import { Label } from '../ui/Label'
import { Spinner } from '../ui/Spinner'
import { browseDirectory, createDirectory } from '../../services/workspace'

interface DirectoryBrowserProps {
  onSelect: (path: string) => void
  initialPath?: string | null
}

export function DirectoryBrowser({ onSelect, initialPath }: DirectoryBrowserProps) {
  const [currentPath, setCurrentPath] = useState<string | null>(initialPath || null)
  const [showHidden, setShowHidden] = useState(false)
  const [showCreateDialog, setShowCreateDialog] = useState(false)
  const [newFolderName, setNewFolderName] = useState('')
  const [createError, setCreateError] = useState<string | null>(null)
  const queryClient = useQueryClient()

  const {
    data: directoryData,
    isLoading,
    isError,
    error,
  } = useQuery({
    queryKey: ['workspace', 'browse', currentPath],
    queryFn: () => browseDirectory(currentPath),
    retry: false,
  })

  const createDirectoryMutation = useMutation({
    mutationFn: ({ parentPath, name }: { parentPath: string; name: string }) =>
      createDirectory(parentPath, name),
    onSuccess: () => {
      // Refresh the directory listing
      queryClient.invalidateQueries({ queryKey: ['workspace', 'browse', currentPath] })
      setShowCreateDialog(false)
      setNewFolderName('')
      setCreateError(null)
    },
    onError: (err: unknown) => {
      const message =
        err instanceof Error
          ? err.message
          : 'Failed to create directory. Please try again.'
      setCreateError(message)
    },
  })

  const handleNavigate = (path: string) => {
    if (directoryData) {
      const entry = directoryData.entries.find((e) => e.path === path)
      if (entry?.is_directory) {
        setCurrentPath(path)
      }
    }
  }

  const handleSelect = (path: string) => {
    onSelect(path)
  }

  const handleGoUp = () => {
    if (directoryData?.parent) {
      setCurrentPath(directoryData.parent)
    }
  }

  const handleGoHome = () => {
    setCurrentPath(null) // null means home directory
  }

  const handleCreateFolder = () => {
    if (!directoryData) return
    setNewFolderName('')
    setCreateError(null)
    setShowCreateDialog(true)
  }

  const handleSubmitCreateFolder = (e: React.FormEvent) => {
    e.preventDefault()
    if (!directoryData || !newFolderName.trim()) {
      setCreateError('Folder name is required')
      return
    }
    createDirectoryMutation.mutate({
      parentPath: directoryData.path,
      name: newFolderName.trim(),
    })
  }

  // Filter directories based on showHidden preference
  const allDirectories = directoryData?.entries.filter((e) => e.is_directory) || []
  const directories = showHidden
    ? allDirectories
    : allDirectories.filter((e) => !e.name.startsWith('.'))
  const files = directoryData?.entries.filter((e) => !e.is_directory) || []

  return (
    <Card className="w-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium">Browse Directory</CardTitle>
          <div className="flex items-center gap-2">
            {directoryData?.parent && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleGoUp}
                className="h-8 px-2"
                title="Go up one level"
              >
                <ChevronUp className="h-4 w-4" />
              </Button>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={handleGoHome}
              className="h-8 px-2"
              title="Go to home directory"
            >
              <Home className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowHidden(!showHidden)}
              className="h-8 px-2"
              title={showHidden ? 'Hide hidden folders' : 'Show hidden folders'}
            >
              {showHidden ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </Button>
            {directoryData && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleCreateFolder}
                className="h-8 px-2"
                title="Create new folder"
              >
                <FolderPlus className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
        <p className="text-xs text-muted-foreground font-mono break-all">
          {directoryData?.path || 'Loading...'}
        </p>
      </CardHeader>
      <CardContent className="flex flex-col min-h-0 max-h-96 p-0">
        {/* Scrollable directory list area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {isLoading ? (
            <div className="space-y-2">
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
            </div>
          ) : isError ? (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription className="text-xs">
                {error instanceof Error ? error.message : 'Failed to browse directory'}
              </AlertDescription>
            </Alert>
          ) : (
            <>
              {/* Directories first */}
              {directories.length === 0 && files.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-4">Directory is empty</p>
              ) : (
                <>
                  {directories.map((entry) => (
                    <button
                      key={entry.path}
                      type="button"
                      onClick={() => handleNavigate(entry.path)}
                      className="w-full flex items-center gap-2 px-3 py-2 rounded-md hover:bg-accent text-left transition-colors"
                    >
                      <Folder className="h-5 w-5 text-blue-500 flex-shrink-0" />
                      <span className="flex-1 text-sm truncate">{entry.name}</span>
                      <ChevronRight className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                    </button>
                  ))}

                  {/* Files (optional, for reference) */}
                  {files.length > 0 && (
                    <div className="pt-2 border-t">
                      <p className="text-xs text-muted-foreground mb-2">Files (for reference)</p>
                      {files.map((entry) => (
                        <div
                          key={entry.path}
                          className="flex items-center gap-2 px-3 py-1 text-sm text-muted-foreground"
                        >
                          <File className="h-4 w-4 flex-shrink-0" />
                          <span className="truncate">{entry.name}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}
            </>
          )}
        </div>

        {/* Fixed select button at bottom */}
        {directoryData && !isLoading && !isError && (
          <div className="border-t p-4 flex-shrink-0">
            <Button
              type="button"
              onClick={() => handleSelect(directoryData.path)}
              className="w-full"
              variant="default"
            >
              <FolderOpen className="h-4 w-4 mr-2" />
              Select This Directory
            </Button>
          </div>
        )}
      </CardContent>

      {/* Create folder dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Create New Folder</DialogTitle>
            <DialogDescription>
              Create a new folder in the current directory: {directoryData?.path}
            </DialogDescription>
          </DialogHeader>
          <form onSubmit={handleSubmitCreateFolder}>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="folder-name">Folder Name</Label>
                <Input
                  id="folder-name"
                  type="text"
                  value={newFolderName}
                  onChange={(e) => {
                    setNewFolderName(e.target.value)
                    setCreateError(null)
                  }}
                  placeholder="my-new-folder"
                  disabled={createDirectoryMutation.isPending}
                  autoFocus
                />
                <p className="text-xs text-muted-foreground">
                  Enter a folder name. The folder will be created in the current directory.
                </p>
              </div>

              {createError && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription className="text-xs">{createError}</AlertDescription>
                </Alert>
              )}
            </div>
            <DialogFooter className="mt-4">
              <Button
                type="button"
                variant="outline"
                onClick={() => {
                  setShowCreateDialog(false)
                  setNewFolderName('')
                  setCreateError(null)
                }}
                disabled={createDirectoryMutation.isPending}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={createDirectoryMutation.isPending || !newFolderName.trim()}>
                {createDirectoryMutation.isPending ? (
                  <>
                    <Spinner className="mr-2 h-4 w-4" />
                    Creating...
                  </>
                ) : (
                  <>
                    <FolderPlus className="mr-2 h-4 w-4" />
                    Create Folder
                  </>
                )}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </Card>
  )
}

