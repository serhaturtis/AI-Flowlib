import { UseQueryResult } from '@tanstack/react-query'
import { FolderOpen, AlertCircle, Plus } from 'lucide-react'
import { ResourceConfigSummary } from '../../../services/configs'
import { Card, CardContent } from '../../ui/Card'
import { Alert, AlertDescription } from '../../ui/Alert'
import { Badge } from '../../ui/Badge'
import { Skeleton } from '../../ui/Skeleton'
import { Button } from '../../ui/Button'

interface ResourceConfigListProps {
  resourcesQuery: UseQueryResult<{ configs: ResourceConfigSummary[] }, Error>
  onSelectResource: (config: ResourceConfigSummary) => void
  onCreateResource: () => void
  selectedProject: string
}

/**
 * Displays list of resource configurations with loading/error states.
 * Shows resource name and resource type.
 */
export function ResourceConfigList({
  resourcesQuery,
  onSelectResource,
  onCreateResource,
  selectedProject,
}: ResourceConfigListProps) {
  if (resourcesQuery.isLoading) {
    return (
      <div className="space-y-2">
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-12 w-full" />
      </div>
    )
  }

  if (resourcesQuery.isError) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>{resourcesQuery.error.message}</AlertDescription>
      </Alert>
    )
  }

  if (!resourcesQuery.data?.configs.length) {
    return (
      <div className="text-center py-8">
        <FolderOpen className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
        <p className="text-muted-foreground text-sm">No resource configs found.</p>
        <Button size="sm" className="mt-4" disabled={!selectedProject} onClick={onCreateResource}>
          <Plus className="h-4 w-4 mr-1" />
          Create Resource
        </Button>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {resourcesQuery.data.configs.map((config) => (
        <Card
          key={config.name}
          className="cursor-pointer hover:bg-accent transition-colors"
          onClick={() => onSelectResource(config)}
        >
          <CardContent className="p-3">
            <div className="font-medium">{config.name}</div>
            <Badge variant="outline" className="text-xs mt-2">
              {config.resource_type}
            </Badge>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
