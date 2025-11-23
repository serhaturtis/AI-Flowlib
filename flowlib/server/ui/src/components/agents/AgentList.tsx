import { Bot, AlertCircle } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../ui/Card'
import { Alert, AlertDescription } from '../ui/Alert'
import { Badge } from '../ui/Badge'
import { Skeleton } from '../ui/Skeleton'

export interface AgentListProps {
  agents: string[] | undefined
  isLoading: boolean
  isError: boolean
  error: Error | null
}

/**
 * Display list of available agents.
 */
export function AgentList({ agents, isLoading, isError, error }: AgentListProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Agent List</CardTitle>
        <CardDescription>Available agents in the selected project</CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-2">
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
          </div>
        ) : isError ? (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error?.message}</AlertDescription>
          </Alert>
        ) : agents && agents.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {agents.map((agent) => (
              <Badge key={agent} variant="secondary" className="text-sm px-3 py-1">
                <Bot className="h-3 w-3 mr-1" />
                {agent}
              </Badge>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <Bot className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground">No agents found.</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
