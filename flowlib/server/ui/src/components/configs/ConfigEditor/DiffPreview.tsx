import type { ConfigDiffResponse } from '../../../services/configs'

export interface DiffPreviewProps {
  diffResult: ConfigDiffResponse | null
}

/**
 * Display component for diff preview with syntax highlighting.
 */
export function DiffPreview({ diffResult }: DiffPreviewProps) {
  if (!diffResult || diffResult.diff.length === 0) {
    return null
  }

  return (
    <pre
      style={{
        backgroundColor: '#272822',
        color: '#f8f8f2',
        padding: '1rem',
        borderRadius: 4,
        overflowX: 'auto',
        marginTop: '1rem',
      }}
    >
      {diffResult.diff.join('')}
    </pre>
  )
}
