import { UseQueryResult } from '@tanstack/react-query'
import { Settings, AlertCircle, Plus } from 'lucide-react'
import { ProviderConfigSummary } from '../../../services/configs'
import { Card } from '../../ui/Card'
import { Alert, AlertDescription } from '../../ui/Alert'
import { Badge } from '../../ui/Badge'
import { Skeleton } from '../../ui/Skeleton'
import { Button } from '../../ui/Button'

interface ProviderConfigListProps {
  providersQuery: UseQueryResult<{ configs: ProviderConfigSummary[] }, Error>
  onSelectProvider: (config: ProviderConfigSummary) => void
  onCreateProvider: () => void
  selectedProject: string
}

/**
 * Displays list of provider configurations with loading/error states.
 * Shows provider name, resource type, and provider type.
 */
export function ProviderConfigList({
  providersQuery,
  onSelectProvider,
  onCreateProvider,
  selectedProject,
}: ProviderConfigListProps) {
  if (providersQuery.isLoading) {
    return (
      <div className="space-y-2">
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-12 w-full" />
      </div>
    )
  }

  if (providersQuery.isError) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>{providersQuery.error.message}</AlertDescription>
      </Alert>
    )
  }

  if (!providersQuery.data?.configs.length) {
    return (
      <div className="text-center py-8">
        <Settings className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
        <p className="text-muted-foreground text-sm">No provider configs found.</p>
        <Button size="sm" className="mt-4" disabled={!selectedProject} onClick={onCreateProvider}>
          <Plus className="h-4 w-4 mr-1" />
          Create Provider
        </Button>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {providersQuery.data.configs.map((config) => (
        <Card
          key={config.name}
          className="cursor-pointer hover:bg-accent transition-colors p-3"
          onClick={() => onSelectProvider(config)}
        >
          <div className="font-medium">{config.name}</div>
          <div className="flex gap-2 mt-2">
            <Badge variant="outline" className="text-xs">
              {config.resource_type}
            </Badge>
            <Badge variant="secondary" className="text-xs">
              {config.provider_type}
            </Badge>
          </div>
        </Card>
      ))}
    </div>
  )
}
