import { UseQueryResult } from '@tanstack/react-query'
import { Bot, AlertCircle } from 'lucide-react'
import { AgentConfigSummary } from '../../../services/configs'
import { Card, CardContent } from '../../ui/Card'
import { Alert, AlertDescription } from '../../ui/Alert'
import { Badge } from '../../ui/Badge'
import { Skeleton } from '../../ui/Skeleton'

interface AgentConfigListProps {
  agentsQuery: UseQueryResult<{ configs: AgentConfigSummary[] }, Error>
  onSelectAgent: (name: string) => void
}

/**
 * Displays list of agent configurations with loading/error states.
 * Shows agent name, persona, and allowed tool categories.
 */
export function AgentConfigList({ agentsQuery, onSelectAgent }: AgentConfigListProps) {
  if (agentsQuery.isLoading) {
    return (
      <div className="space-y-2">
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-12 w-full" />
      </div>
    )
  }

  if (agentsQuery.isError) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>{agentsQuery.error.message}</AlertDescription>
      </Alert>
    )
  }

  if (!agentsQuery.data?.configs.length) {
    return (
      <div className="text-center py-8">
        <Bot className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
        <p className="text-muted-foreground text-sm">No agent configs found.</p>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {agentsQuery.data.configs.map((agent) => (
        <Card
          key={agent.name}
          className="cursor-pointer hover:bg-accent transition-colors"
          onClick={() => onSelectAgent(agent.name)}
        >
          <CardContent className="p-3">
            <div className="font-medium">{agent.name}</div>
            <div className="text-sm text-muted-foreground mt-1 line-clamp-1">{agent.persona}</div>
            {agent.allowed_tool_categories.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-2">
                {agent.allowed_tool_categories.map((cat) => (
                  <Badge key={cat} variant="secondary" className="text-xs">
                    {cat}
                  </Badge>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
