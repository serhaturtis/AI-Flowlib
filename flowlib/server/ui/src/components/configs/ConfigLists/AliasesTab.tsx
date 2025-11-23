import { useState, useEffect } from 'react'
import { UseQueryResult, useQueryClient } from '@tanstack/react-query'
import { Plus, AlertCircle, CheckCircle2 } from 'lucide-react'
import { applyAliases } from '../../../services/configs'
import { Card, CardContent } from '../../ui/Card'
import { Alert, AlertDescription } from '../../ui/Alert'
import { Skeleton } from '../../ui/Skeleton'
import { Button } from '../../ui/Button'
import { Input } from '../../ui/Input'
import { Spinner } from '../../ui/Spinner'
import { Stack } from '../../layout/Stack'

interface Alias {
  alias: string
  canonical: string
}

interface AliasesTabProps {
  aliasesQuery: UseQueryResult<{ aliases: Alias[] }, Error>
  selectedProject: string
}

/**
 * Tab component for managing alias bindings.
 * Allows adding, removing, and saving alias configurations.
 */
export function AliasesTab({ aliasesQuery, selectedProject }: AliasesTabProps) {
  const queryClient = useQueryClient()
  const [aliasesDraft, setAliasesDraft] = useState<Alias[]>([])
  const [newAlias, setNewAlias] = useState('')
  const [newCanonical, setNewCanonical] = useState('')
  const [aliasSaveError, setAliasSaveError] = useState<string | null>(null)
  const [aliasSaveSuccess, setAliasSaveSuccess] = useState<string | null>(null)
  const [isSavingAliases, setIsSavingAliases] = useState(false)

  // Sync draft with query data
  useEffect(() => {
    if (aliasesQuery.data?.aliases) {
      setAliasesDraft(
        aliasesQuery.data.aliases.map((a) => ({ alias: a.alias, canonical: a.canonical })),
      )
    } else {
      setAliasesDraft([])
    }
  }, [aliasesQuery.data?.aliases])

  const removeAliasAt = (idx: number) => {
    setAliasesDraft((prev) => prev.filter((_, i) => i !== idx))
  }

  const addAliasRow = () => {
    if (!newAlias.trim() || !newCanonical.trim()) return
    setAliasesDraft((prev) => [...prev, { alias: newAlias.trim(), canonical: newCanonical.trim() }])
    setNewAlias('')
    setNewCanonical('')
  }

  const handleApplyAliases = async () => {
    if (!selectedProject) return
    setAliasSaveError(null)
    setAliasSaveSuccess(null)

    try {
      setIsSavingAliases(true)
      await applyAliases({
        project_id: selectedProject,
        aliases: aliasesDraft,
      })
      await queryClient.invalidateQueries({ queryKey: ['configs', 'aliases', selectedProject] })
      setAliasSaveSuccess('Aliases updated successfully.')
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to update aliases.'
      setAliasSaveError(message)
    } finally {
      setIsSavingAliases(false)
    }
  }

  if (aliasesQuery.isLoading) {
    return (
      <div className="space-y-2">
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-12 w-full" />
      </div>
    )
  }

  if (aliasesQuery.isError) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>{aliasesQuery.error.message}</AlertDescription>
      </Alert>
    )
  }

  return (
    <>
      {aliasesDraft.length ? (
        <div className="space-y-2 mb-4">
          {aliasesDraft.map((alias, idx) => (
            <Card key={`${alias.alias}-${idx}`}>
              <CardContent className="p-3 flex items-center justify-between">
                <div className="flex gap-2">
                  <code className="bg-muted px-2 py-1 rounded text-sm">{alias.alias}</code>
                  <span className="text-muted-foreground">→</span>
                  <code className="bg-muted px-2 py-1 rounded text-sm">{alias.canonical}</code>
                </div>
                <Button variant="ghost" size="sm" onClick={() => removeAliasAt(idx)}>
                  Remove
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <div className="text-center py-8 mb-4">
          <p className="text-muted-foreground text-sm">No aliases defined.</p>
        </div>
      )}

      <Stack spacing="sm">
        <div className="grid grid-cols-[1fr_1fr_auto] gap-2">
          <Input
            type="text"
            placeholder="alias (e.g., default-llm)"
            value={newAlias}
            onChange={(e) => setNewAlias(e.target.value)}
          />
          <Input
            type="text"
            placeholder="canonical (e.g., example-llamacpp-provider)"
            value={newCanonical}
            onChange={(e) => setNewCanonical(e.target.value)}
          />
          <Button type="button" onClick={addAliasRow}>
            <Plus className="h-4 w-4" />
          </Button>
        </div>

        <Button
          type="button"
          onClick={handleApplyAliases}
          disabled={isSavingAliases || !selectedProject}
          className="w-full"
        >
          {isSavingAliases ? (
            <>
              <Spinner size="sm" className="mr-2" />
              Saving…
            </>
          ) : (
            'Apply Aliases'
          )}
        </Button>

        {aliasSaveError && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{aliasSaveError}</AlertDescription>
          </Alert>
        )}

        {aliasSaveSuccess && (
          <Alert variant="success">
            <CheckCircle2 className="h-4 w-4" />
            <AlertDescription>{aliasSaveSuccess}</AlertDescription>
          </Alert>
        )}
      </Stack>
    </>
  )
}
