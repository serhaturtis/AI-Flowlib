/**
 * Plugin Grid View Component
 *
 * Displays knowledge plugins in a responsive grid layout with cards.
 */

import { Brain, Trash2, Eye, Database, Network } from 'lucide-react'
import { PluginSummary } from '../../services/knowledge'
import { Button } from '../ui/Button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/Card'
import { Badge } from '../ui/Badge'

type Props = {
  plugins: PluginSummary[]
  onDelete: (pluginId: string) => void
  onViewDetails: (pluginId: string) => void
  isDeleting: boolean
}

export function PluginGridView({ plugins, onDelete, onViewDetails, isDeleting }: Props) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {plugins.map((plugin) => (
        <Card key={plugin.plugin_id} className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-primary" />
                <CardTitle className="text-base">{plugin.name}</CardTitle>
              </div>
              <div className="flex gap-1">
                {plugin.capabilities.has_vector_db && (
                  <span title="Vector DB">
                    <Database className="h-4 w-4 text-muted-foreground" />
                  </span>
                )}
                {plugin.capabilities.has_graph_db && (
                  <span title="Graph DB">
                    <Network className="h-4 w-4 text-muted-foreground" />
                  </span>
                )}
              </div>
            </div>
            <CardDescription className="line-clamp-2">
              {plugin.description || 'No description provided'}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Domains */}
            <div className="flex flex-wrap gap-1">
              {plugin.domains.slice(0, 3).map((domain) => (
                <Badge key={domain} variant="secondary" className="text-xs">
                  {domain}
                </Badge>
              ))}
              {plugin.domains.length > 3 && (
                <Badge variant="secondary" className="text-xs">
                  +{plugin.domains.length - 3}
                </Badge>
              )}
            </div>

            {/* Stats */}
            <div className="grid grid-cols-3 gap-2 text-center">
              <div>
                <div className="text-2xl font-bold">{plugin.extraction_stats.total_entities}</div>
                <div className="text-xs text-muted-foreground">Entities</div>
              </div>
              <div>
                <div className="text-2xl font-bold">{plugin.extraction_stats.total_relationships}</div>
                <div className="text-xs text-muted-foreground">Relations</div>
              </div>
              <div>
                <div className="text-2xl font-bold">{plugin.extraction_stats.total_documents}</div>
                <div className="text-xs text-muted-foreground">Documents</div>
              </div>
            </div>

            {/* Metadata */}
            <div className="text-xs text-muted-foreground space-y-1">
              <div>Version: {plugin.version}</div>
              <div>Author: {plugin.author}</div>
              <div>Created: {new Date(plugin.created_at).toLocaleDateString()}</div>
            </div>

            {/* Actions */}
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => onViewDetails(plugin.plugin_id)}
                className="flex-1"
              >
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
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
