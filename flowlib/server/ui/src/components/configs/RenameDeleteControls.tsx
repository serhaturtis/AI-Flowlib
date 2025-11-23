import { useState } from 'react'
import { renameConfig, deleteConfig } from '../../services/configs'
import { Button } from '../ui/Button'
import { Input } from '../ui/Input'
import { Label } from '../ui/Label'

interface RenameDeleteControlsProps {
  projectId: string
  currentPath: string
  onRenamed: () => void
  onDeleted: () => void
  onError: (msg: string) => void
}

/**
 * Component for renaming and deleting configuration files.
 * Provides UI controls with confirmation and error handling.
 */
export function RenameDeleteControls({
  projectId,
  currentPath,
  onRenamed,
  onDeleted,
  onError,
}: RenameDeleteControlsProps) {
  const [newPath, setNewPath] = useState(currentPath)
  const [busy, setBusy] = useState(false)

  const doRename = async () => {
    if (!newPath.trim() || newPath.trim() === currentPath) {
      onError('Enter a different target path.')
      return
    }

    try {
      setBusy(true)
      await renameConfig({
        project_id: projectId,
        old_relative_path: currentPath,
        new_relative_path: newPath.trim(),
      })
      onRenamed()
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Rename failed.'
      onError(msg)
    } finally {
      setBusy(false)
    }
  }

  const doDelete = async () => {
    if (!confirm(`Delete ${currentPath}? This action cannot be undone.`)) {
      return
    }

    try {
      setBusy(true)
      await deleteConfig({
        project_id: projectId,
        relative_path: currentPath,
      })
      onDeleted()
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Delete failed.'
      onError(msg)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="grid gap-2">
      <div className="grid gap-2">
        <Label htmlFor="rename-new-path">New Path</Label>
        <Input
          id="rename-new-path"
          type="text"
          value={newPath}
          onChange={(e) => setNewPath(e.target.value)}
          placeholder="configs/providers/new_name.py"
        />
      </div>
      <div className="flex gap-2">
        <Button type="button" onClick={doRename} disabled={busy} variant="outline" size="sm">
          {busy ? 'Renaming…' : 'Rename'}
        </Button>
        <Button type="button" onClick={doDelete} disabled={busy} variant="destructive" size="sm">
          {busy ? 'Deleting…' : 'Delete'}
        </Button>
      </div>
    </div>
  )
}
