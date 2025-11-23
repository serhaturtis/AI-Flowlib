import { FormEvent } from 'react'
import { Button } from '../../ui/Button'
import { Textarea } from '../../ui/Textarea'
import type { UseConfigMutationsResult } from '../../../hooks/configs/useConfigMutations'

export interface RawEditorFormProps {
  content: string
  onContentChange: (value: string) => void
  diffMutation: UseConfigMutationsResult['diffMutation']
  applyMutation: UseConfigMutationsResult['applyMutation']
  onDiff: () => void
  onApply: () => void
}

/**
 * Raw mode editor form with textarea and diff/apply buttons.
 */
export function RawEditorForm({
  content,
  onContentChange,
  diffMutation,
  applyMutation,
  onDiff,
  onApply,
}: RawEditorFormProps) {
  const handleDiff = (event: FormEvent) => {
    event.preventDefault()
    onDiff()
  }

  const handleApply = (event: FormEvent) => {
    event.preventDefault()
    onApply()
  }

  return (
    <form className="grid gap-4">
      <Textarea
        value={content}
        onChange={(event) => onContentChange(event.target.value)}
        rows={16}
        className="font-mono"
      />
      <div className="flex gap-3">
        <Button type="button" onClick={handleDiff} disabled={diffMutation.isPending}>
          {diffMutation.isPending ? 'Diffing…' : 'Show Diff'}
        </Button>
        <Button type="button" onClick={handleApply} disabled={applyMutation.isPending} variant="secondary">
          {applyMutation.isPending ? 'Applying…' : 'Apply'}
        </Button>
      </div>
    </form>
  )
}
