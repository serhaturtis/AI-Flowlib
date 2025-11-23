import { useState, useEffect } from 'react'
import { Plus, AlertCircle, CheckCircle2 } from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/Dialog'
import { Button } from '../ui/Button'
import { Input } from '../ui/Input'
import { Select } from '../ui/Select'
import { Alert, AlertDescription } from '../ui/Alert'
import { Spinner } from '../ui/Spinner'
import { FormField } from '../forms/FormField'
import { ProviderConfigSummary, ResourceConfigSummary, ConfigType } from '../../services/configs'
import { useProviderFormState } from '../../hooks/configs/useProviderFormState'
import { useResourceFormState } from '../../hooks/configs/useResourceFormState'
import { useCreateMutation } from '../../hooks/configs/useCreateMutation'
import { ProviderFormFields } from './CreateConfigDialog/ProviderFormFields'
import { ResourceFormFields } from './CreateConfigDialog/ResourceFormFields'

export interface CreateConfigDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  selectedProject: string
  initialType?: ConfigType
  onSuccess: (result: {
    type: ConfigType
    name: string
    config: ProviderConfigSummary | ResourceConfigSummary
  }) => void
}

/**
 * Dialog for creating new provider or resource configurations.
 *
 * Features:
 * - Dual-mode form (provider vs resource)
 * - Dynamic schema loading based on resource/config type
 * - JSON validation
 * - Comprehensive error handling with server validation details
 * - Auto-navigation to created config on success
 */
export function CreateConfigDialog({
  open,
  onOpenChange,
  selectedProject,
  initialType = 'provider',
  onSuccess,
}: CreateConfigDialogProps) {
  const [createType, setCreateType] = useState<ConfigType>(initialType)
  const [createName, setCreateName] = useState('')

  // Sync createType with initialType when dialog opens
  useEffect(() => {
    if (open) {
      setCreateType(initialType)
    }
  }, [open, initialType])

  // Custom hooks for state management
  const providerState = useProviderFormState(selectedProject, createType === 'provider')
  const resourceState = useResourceFormState(selectedProject, createType === 'resource')
  const mutation = useCreateMutation()

  // Reset all form fields
  const resetForm = () => {
    setCreateName('')
    mutation.setError(null)
    mutation.setSuccess(null)
    if (createType === 'provider') {
      providerState.resetForm()
    } else {
      resourceState.resetForm()
    }
  }

  // Handle config creation submission
  const handleCreate = () => {
    mutation.create({
      type: createType,
      projectId: selectedProject,
      name: createName,
      providerResourceType: providerState.resourceType,
      providerType: providerState.providerType,
      providerDescription: providerState.description,
      providerSettingsJson: providerState.settingsJson,
      resourceType: resourceState.resourceType,
      resourceProviderType: resourceState.providerType,
      resourceDescription: resourceState.description,
      resourceConfigJson: resourceState.configJson,
      onSuccess,
      onClose: () => onOpenChange(false),
      resetForm,
    })
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(isOpen) => {
        onOpenChange(isOpen)
        if (!isOpen) {
          resetForm()
        }
      }}
    >
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto w-[95vw] sm:w-full">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Plus className="h-5 w-5" />
            Create Configuration
          </DialogTitle>
          <DialogDescription>Create a new provider or resource configuration</DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <FormField label="Type">
            <Select value={createType} onChange={(event) => setCreateType(event.target.value as ConfigType)}>
              <option value="provider">Provider Config</option>
              <option value="resource">Model Resource Config</option>
            </Select>
          </FormField>

          <FormField label="Canonical Name" required>
            <Input
              type="text"
              value={createName}
              onChange={(event) => setCreateName(event.target.value)}
              placeholder="example-provider"
            />
          </FormField>

          {createType === 'provider' ? (
            <ProviderFormFields providerState={providerState} />
          ) : (
            <ResourceFormFields resourceState={resourceState} />
          )}

          {mutation.error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{mutation.error}</AlertDescription>
            </Alert>
          )}

          {mutation.success && (
            <Alert variant="success">
              <CheckCircle2 className="h-4 w-4" />
              <AlertDescription>{mutation.success}</AlertDescription>
            </Alert>
          )}
        </div>

        <DialogFooter>
          <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            type="button"
            onClick={handleCreate}
            disabled={
              mutation.isCreating ||
              !selectedProject ||
              !createName.trim() ||
              (createType === 'provider'
                ? !providerState.providerType.trim()
                : !resourceState.providerType.trim())
            }
          >
            {mutation.isCreating ? (
              <>
                <Spinner size="sm" className="mr-2" />
                Creating…
              </>
            ) : (
              <>
                <Plus className="h-4 w-4 mr-2" />
                Create Configuration
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
