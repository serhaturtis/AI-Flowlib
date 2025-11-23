import { useState } from 'react'
import { Terminal, Send, Copy, Trash2 } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '../ui/Card'
import { Button } from '../ui/Button'
import { Textarea } from '../ui/Textarea'
import { Separator } from '../ui/Separator'
import { Stack } from '../layout/Stack'
import type { UseReplSessionResult } from '../../hooks/agents/useReplSession'

export interface ReplSessionProps {
  repl: UseReplSessionResult
  selectedProject: string
  agentName: string
}

/**
 * Interactive REPL session interface.
 *
 * Features:
 * - Start/close REPL sessions
 * - Real-time event log via WebSocket
 * - Message sending
 * - Copy/clear log functionality
 * - Auto-scrolling
 * - Syntax-highlighted output
 */
export function ReplSession({ repl, selectedProject, agentName }: ReplSessionProps) {
  const { session, messages, logRef, startSession, closeSession, sendMessage, clearMessages, error } =
    repl
  const [input, setInput] = useState('')

  const handleSend = async () => {
    if (!input.trim()) return
    try {
      await sendMessage(input)
      setInput('')
    } catch (err) {
      // Error already handled by hook
    }
  }

  const handleCopyLog = () => {
    const text = messages
      .filter((m) => m.content || m.message) // Skip messages without content
      .map((m) => `[${m.ts ?? ''}] ${m.type.toUpperCase()}: ${m.content ?? m.message ?? '(no content)'}`)
      .join('\n')
    void navigator.clipboard.writeText(text)
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Terminal className="h-5 w-5" />
          REPL Session
        </CardTitle>
        <CardDescription>Interactive REPL session for agent communication</CardDescription>
      </CardHeader>
      <CardContent>
        {session ? (
          <Stack spacing="md">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-sm text-muted-foreground">Session:</span>{' '}
                <code className="bg-muted px-2 py-1 rounded text-xs">{session.session_id}</code>
              </div>
              <Button type="button" variant="outline" size="sm" onClick={() => closeSession()}>
                Close Session
              </Button>
            </div>

            <Separator />

            <div className="border border-border rounded-lg overflow-hidden">
              <div className="bg-muted px-4 py-2 flex items-center justify-between border-b border-border">
                <span className="text-sm font-medium">Event Log</span>
                <div className="flex gap-2">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={clearMessages}
                    title="Clear output"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={handleCopyLog}
                    title="Copy output"
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              <div
                ref={logRef}
                className="bg-[#1e1e1e] text-[#d4d4d4] p-4 font-mono text-sm max-h-[500px] overflow-y-auto scrollbar-thin"
                style={{ minHeight: '300px' }}
              >
                {messages.length === 0 ? (
                  <div className="text-muted-foreground">Waiting for events...</div>
                ) : (
                  messages.map((msg, idx) => {
                    const isAgent = msg.type === 'agent'
                    const isUser = msg.type === 'user'
                    const isSystem = msg.type === 'system'
                    return (
                      <div key={idx} className="mb-2">
                        <span className="text-[#858585] mr-2">
                          {msg.ts ? new Date(msg.ts).toLocaleTimeString() : ''}
                        </span>
                        <span
                          className={
                            isAgent
                              ? 'text-[#4ec9b0]'
                              : isUser
                                ? 'text-[#569cd6]'
                                : isSystem
                                  ? 'text-[#ce9178]'
                                  : 'text-[#d4d4d4]'
                          }
                        >
                          <strong>
                            {isAgent ? 'Agent' : isUser ? 'You' : isSystem ? 'System' : 'Unknown'}:
                          </strong>{' '}
                          {msg.content ?? msg.message ?? '(no content)'}
                        </span>
                      </div>
                    )
                  })
                )}
              </div>
            </div>

            <div className="flex gap-2">
              <Textarea
                rows={3}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                    e.preventDefault()
                    void handleSend()
                  }
                }}
                placeholder="Type your message... (Ctrl/Cmd+Enter to send)"
                className="flex-1 font-mono"
              />
              <Button type="button" onClick={handleSend} disabled={!input} className="self-end">
                <Send className="h-4 w-4 mr-2" />
                Send
              </Button>
            </div>

            {error && (
              <div className="text-sm text-destructive">
                Error: {error}
              </div>
            )}
          </Stack>
        ) : (
          <Stack spacing="md">
            <p className="text-muted-foreground">
              Start a REPL session with the selected project and agent.
            </p>
            <Button
              type="button"
              onClick={() => startSession(selectedProject, agentName)}
              disabled={!selectedProject || !agentName}
              className="w-full sm:w-auto"
            >
              <Terminal className="h-4 w-4 mr-2" />
              Start REPL Session
            </Button>

            {error && (
              <div className="text-sm text-destructive">
                Error: {error}
              </div>
            )}
          </Stack>
        )}
      </CardContent>
    </Card>
  )
}
