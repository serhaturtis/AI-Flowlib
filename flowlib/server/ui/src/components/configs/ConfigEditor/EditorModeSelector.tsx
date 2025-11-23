export interface EditorModeSelectorProps {
  mode: 'raw' | 'structured'
  onModeChange: (mode: 'raw' | 'structured') => void
}

/**
 * Selector for toggling between raw and structured editor modes.
 */
export function EditorModeSelector({ mode, onModeChange }: EditorModeSelectorProps) {
  return (
    <div style={{ marginBottom: '0.5rem' }}>
      <label style={{ display: 'inline-flex', gap: '0.5rem', alignItems: 'center' }}>
        <span>Editor:</span>
        <select value={mode} onChange={(e) => onModeChange(e.target.value as 'raw' | 'structured')}>
          <option value="raw">Raw</option>
          <option value="structured">Structured</option>
        </select>
      </label>
    </div>
  )
}
