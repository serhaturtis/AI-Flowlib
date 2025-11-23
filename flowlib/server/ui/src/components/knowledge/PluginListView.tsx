/**
 * Plugin List View Component
 *
 * Displays knowledge plugins in a table/list format.
 */

import { Brain, Trash2, Eye, Database, Network } from 'lucide-react'
import { PluginSummary } from '../../services/knowledge'
import { Button } from '../ui/Button'
import { Badge } from '../ui/Badge'

type Props = {
  plugins: PluginSummary[]
  onDelete: (pluginId: string) => void
  onViewDetails: (pluginId: string) => void
  isDeleting: boolean
}

export function PluginListView({ plugins, onDelete, onViewDetails, isDeleting }: Props) {
  return (
    <div className="space-y-2">
      {plugins.map((plugin) => (
        <div
          key={plugin.plugin_id}
          className="flex items-center gap-4 p-4 border rounded-lg hover:bg-muted/50 transition-colors"
        >
          {/* Icon */}
          <Brain className="h-8 w-8 text-primary flex-shrink-0" />

          {/* Main Info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <h3 className="font-semibold truncate">{plugin.name}</h3>
              <span className="text-xs text-muted-foreground">v{plugin.version}</span>
              {plugin.capabilities.has_vector_db && (
                <span title="Vector DB">
                  <Database className="h-3 w-3 text-muted-foreground" />
                </span>
              )}
              {plugin.capabilities.has_graph_db && (
                <span title="Graph DB">
                  <Network className="h-3 w-3 text-muted-foreground" />
                </span>
              )}
            </div>
            <p className="text-sm text-muted-foreground line-clamp-1">
              {plugin.description || 'No description provided'}
            </p>
            <div className="flex flex-wrap gap-1 mt-2">
              {plugin.domains.map((domain) => (
                <Badge key={domain} variant="secondary" className="text-xs">
                  {domain}
                </Badge>
              ))}
            </div>
          </div>

          {/* Stats */}
          <div className="hidden md:flex gap-6 text-center">
            <div>
              <div className="text-lg font-bold">{plugin.extraction_stats.total_entities}</div>
              <div className="text-xs text-muted-foreground">Entities</div>
            </div>
            <div>
              <div className="text-lg font-bold">{plugin.extraction_stats.total_relationships}</div>
              <div className="text-xs text-muted-foreground">Relations</div>
            </div>
            <div>
              <div className="text-lg font-bold">{plugin.extraction_stats.total_documents}</div>
              <div className="text-xs text-muted-foreground">Docs</div>
            </div>
          </div>

          {/* Created Date */}
          <div className="hidden lg:block text-sm text-muted-foreground text-right">
            {new Date(plugin.created_at).toLocaleDateString()}
          </div>

          {/* Actions */}
          <div className="flex gap-2 flex-shrink-0">
            <Button size="sm" variant="outline" onClick={() => onViewDetails(plugin.plugin_id)}>
              <Eye className="h-4 w-4 mr-1" />
              Details
            </Button>
            <Button
              size="sm"
              variant="destructive"
              onClick={() => onDelete(plugin.plugin_id)}
              disabled={isDeleting}
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      ))}
    </div>
  )
}
