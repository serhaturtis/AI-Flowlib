/**
 * Knowledge Plugin Multi-Select Component
 *
 * Allows selecting multiple knowledge plugins from available options.
 */

import { useState } from 'react'
import { Brain, X } from 'lucide-react'
import { usePluginList } from '../../hooks/knowledge/useKnowledgePlugins'
import { Label } from '../ui/Label'
import { Badge } from '../ui/Badge'
import { Skeleton } from '../ui/Skeleton'
import { Alert, AlertDescription } from '../ui/Alert'

type Props = {
  projectId: string | null
  selectedPluginIds: string[]
  onChange: (pluginIds: string[]) => void
  label?: string
  description?: string
  error?: string
}

export function KnowledgePluginMultiSelect({
  projectId,
  selectedPluginIds,
  onChange,
  label = 'Knowledge Plugins',
  description = 'Select knowledge plugins to attach to this agent',
  error,
}: Props) {
  const { data, isLoading, isError } = usePluginList(projectId)
  const [isOpen, setIsOpen] = useState(false)

  const availablePlugins = data?.plugins || []
  const selectedPlugins = availablePlugins.filter((p) => selectedPluginIds.includes(p.plugin_id))
  const unselectedPlugins = availablePlugins.filter((p) => !selectedPluginIds.includes(p.plugin_id))

  const handleToggle = (pluginId: string) => {
    if (selectedPluginIds.includes(pluginId)) {
      onChange(selectedPluginIds.filter((id) => id !== pluginId))
    } else {
      onChange([...selectedPluginIds, pluginId])
    }
  }

  const handleRemove = (pluginId: string) => {
    onChange(selectedPluginIds.filter((id) => id !== pluginId))
  }

  if (!projectId) {
    return (
      <div className="space-y-2">
        <Label>{label}</Label>
        <Alert>
          <AlertDescription>Please select a project to view available knowledge plugins.</AlertDescription>
        </Alert>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <Label htmlFor="knowledge-plugins">{label}</Label>
      {description && <p className="text-sm text-muted-foreground">{description}</p>}

      {/* Selected Plugins */}
      {selectedPlugins.length > 0 && (
        <div className="flex flex-wrap gap-2 p-3 border rounded-md bg-muted/50">
          {selectedPlugins.map((plugin) => (
            <Badge key={plugin.plugin_id} variant="secondary" className="flex items-center gap-1">
              <Brain className="h-3 w-3" />
              {plugin.name}
              <button
                type="button"
                onClick={() => handleRemove(plugin.plugin_id)}
                className="ml-1 hover:bg-muted-foreground/20 rounded p-0.5"
              >
                <X className="h-3 w-3" />
              </button>
            </Badge>
          ))}
        </div>
      )}

      {/* Dropdown */}
      <div className="relative">
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          className="w-full px-3 py-2 text-left border border-input rounded-md bg-background hover:bg-muted/50 transition-colors"
        >
          {selectedPlugins.length === 0
            ? 'Select plugins...'
            : `${selectedPlugins.length} plugin${selectedPlugins.length === 1 ? '' : 's'} selected`}
        </button>

        {isOpen && (
          <div className="absolute z-10 w-full mt-1 max-h-64 overflow-y-auto border border-input rounded-md bg-background shadow-lg">
            {isLoading && (
              <div className="p-3 space-y-2">
                <Skeleton className="h-8" />
                <Skeleton className="h-8" />
              </div>
            )}

            {isError && (
              <div className="p-3">
                <Alert variant="destructive">
                  <AlertDescription>Failed to load plugins</AlertDescription>
                </Alert>
              </div>
            )}

            {!isLoading && !isError && availablePlugins.length === 0 && (
              <div className="p-3 text-sm text-muted-foreground text-center">
                No knowledge plugins available. Create one first.
              </div>
            )}

            {!isLoading && !isError && availablePlugins.length > 0 && (
              <>
                {unselectedPlugins.length === 0 && selectedPlugins.length > 0 && (
                  <div className="p-3 text-sm text-muted-foreground text-center">All plugins selected</div>
                )}
                {unselectedPlugins.map((plugin) => (
                  <button
                    key={plugin.plugin_id}
                    type="button"
                    onClick={() => handleToggle(plugin.plugin_id)}
                    className="w-full px-3 py-2 text-left hover:bg-muted/50 transition-colors border-b border-border last:border-b-0"
                  >
                    <div className="flex items-start gap-2">
                      <Brain className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm">{plugin.name}</div>
                        <div className="text-xs text-muted-foreground truncate">
                          {plugin.description || 'No description'}
                        </div>
                        <div className="flex gap-1 mt-1">
                          {plugin.domains.slice(0, 2).map((domain) => (
                            <span key={domain} className="text-xs px-1 py-0.5 bg-primary/10 rounded">
                              {domain}
                            </span>
                          ))}
                          {plugin.domains.length > 2 && (
                            <span className="text-xs px-1 py-0.5 bg-primary/10 rounded">
                              +{plugin.domains.length - 2}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </button>
                ))}
              </>
            )}
          </div>
        )}
      </div>

      {error && <span className="text-sm text-destructive">{error}</span>}
    </div>
  )
}
