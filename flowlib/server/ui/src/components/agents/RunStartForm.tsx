import { Play, AlertCircle } from 'lucide-react'
import { UseQueryResult } from '@tanstack/react-query'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../ui/Card'
import { FormField } from '../forms/FormField'
import { Select } from '../ui/Select'
import { Input } from '../ui/Input'
import { Button } from '../ui/Button'
import { Alert, AlertDescription } from '../ui/Alert'
import { Spinner } from '../ui/Spinner'
import { Skeleton } from '../ui/Skeleton'
import { Stack } from '../layout/Stack'
import type { UseAgentRunFormResult } from '../../hooks/agents/useAgentRunForm'

export interface RunStartFormProps {
  form: UseAgentRunFormResult
  agentsQuery: UseQueryResult<string[], Error>
  selectedProject: string
}

/**
 * Form for starting a new agent run.
 *
 * Features:
 * - Agent selection
 * - Execution mode selection
 * - Max cycles configuration (for autonomous mode)
 * - Form validation
 * - Loading states
 */
export function RunStartForm({ form, agentsQuery, selectedProject }: RunStartFormProps) {
  const {
    agentName,
    mode,
    maxCycles,
    formError,
    setAgentName,
    setMode,
    setMaxCycles,
    startMutation,
    handleStart,
  } = form

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Play className="h-5 w-5" />
          Start Run
        </CardTitle>
        <CardDescription>Start a new agent run</CardDescription>
      </CardHeader>
      <CardContent>
        <Stack spacing="lg">
          <FormField label="Agent" required>
            {agentsQuery.isLoading ? (
              <Skeleton className="h-10 w-full" />
            ) : agentsQuery.isError ? (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{agentsQuery.error.message}</AlertDescription>
              </Alert>
            ) : (
              <Select value={agentName} onChange={(event) => setAgentName(event.target.value)}>
                <option value="">Select agent</option>
                {agentsQuery.data?.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </Select>
            )}
          </FormField>

          <FormField label="Mode" required>
            <Select value={mode} onChange={(event) => setMode(event.target.value)}>
              <option value="repl">REPL (Interactive)</option>
              <option value="autonomous">Autonomous</option>
              <option value="daemon">Daemon</option>
              <option value="remote">Remote</option>
            </Select>
          </FormField>

          {mode === 'autonomous' && (
            <FormField label="Max Cycles" description="Maximum number of execution cycles">
              <Input
                type="number"
                value={maxCycles}
                min={1}
                onChange={(event) => setMaxCycles(event.target.value)}
              />
            </FormField>
          )}

          {formError && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{formError}</AlertDescription>
            </Alert>
          )}

          <Button
            type="button"
            onClick={() => handleStart(selectedProject)}
            disabled={startMutation.isPending || !selectedProject || !agentName}
            className="w-full"
          >
            {startMutation.isPending ? (
              <>
                <Spinner size="sm" className="mr-2" />
                Starting…
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Start Run
              </>
            )}
          </Button>
        </Stack>
      </CardContent>
    </Card>
  )
}
