import { ResourceConfigSummary } from '../../services/configs'

/**
 * Result type for JSON parsing operations
 */
export interface JsonParseResult<T = unknown> {
  success: boolean
  data?: T
  error?: string
}

/**
 * Safely stringify a value to JSON with pretty printing.
 * Returns result with error message if stringify fails.
 */
export function stringifyJson(value: unknown): JsonParseResult<string> {
  try {
    return {
      success: true,
      data: JSON.stringify(value, null, 2),
    }
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Failed to stringify JSON',
    }
  }
}

/**
 * Parse JSON string to object with error feedback.
 * Returns result with error message if parsing fails.
 */
export function parseJson<T = Record<string, unknown>>(value: string): JsonParseResult<T> {
  if (!value || !value.trim()) {
    return {
      success: false,
      error: 'JSON input is empty',
    }
  }

  try {
    const parsed = JSON.parse(value)
    return {
      success: true,
      data: parsed as T,
    }
  } catch (error) {
    // Extract helpful error message from SyntaxError
    let errorMsg = 'Invalid JSON'
    if (error instanceof SyntaxError) {
      // Parse out position information if available
      const match = error.message.match(/position (\d+)/)
      if (match) {
        const pos = parseInt(match[1], 10)
        const preview = value.slice(Math.max(0, pos - 20), pos + 20)
        errorMsg = `${error.message}\nNear: "${preview}"`
      } else {
        errorMsg = error.message
      }
    }
    return {
      success: false,
      error: errorMsg,
    }
  }
}


/**
 * Resolve the file path for a resource configuration.
 * TODO: Future improvement - keep actual path in metadata instead of inferring.
 */
export function resolveResourcePath(config: ResourceConfigSummary): string {
  return `configs/resources/${config.name}.py`
}
