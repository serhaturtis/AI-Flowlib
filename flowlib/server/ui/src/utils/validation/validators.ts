/**
 * Validation constraints matching backend API (flowlib_server/app/models/projects.py)
 * These MUST stay in sync with backend validation to ensure consistent behavior.
 */
export const VALIDATION_LIMITS = {
  PROJECT_NAME_MIN_LENGTH: 3,
  PROJECT_NAME_MAX_LENGTH: 128,
  DESCRIPTION_MAX_LENGTH: 2048,
  LIST_ITEM_MAX_LENGTH: 100,
} as const

/**
 * Validation result type
 */
export interface ValidationResult {
  isValid: boolean
  error?: string
}

/**
 * Validator function type
 */
export type Validator<T = string> = (value: T) => ValidationResult

/**
 * Project name validation rules
 * MUST match backend: flowlib_server/app/models/projects.py:19
 */
export const validateProjectName: Validator = (value: string): ValidationResult => {
  const trimmed = value.trim()

  if (!trimmed) {
    return {
      isValid: false,
      error: 'Project name is required',
    }
  }

  if (trimmed.length < VALIDATION_LIMITS.PROJECT_NAME_MIN_LENGTH) {
    return {
      isValid: false,
      error: `Project name must be at least ${VALIDATION_LIMITS.PROJECT_NAME_MIN_LENGTH} characters`,
    }
  }

  if (trimmed.length > VALIDATION_LIMITS.PROJECT_NAME_MAX_LENGTH) {
    return {
      isValid: false,
      error: `Project name must not exceed ${VALIDATION_LIMITS.PROJECT_NAME_MAX_LENGTH} characters`,
    }
  }

  // Allow alphanumeric, hyphens, underscores, and spaces
  const validPattern = /^[a-zA-Z0-9\s_-]+$/
  if (!validPattern.test(trimmed)) {
    return {
      isValid: false,
      error: 'Project name can only contain letters, numbers, spaces, hyphens, and underscores',
    }
  }

  // Must start with alphanumeric
  if (!/^[a-zA-Z0-9]/.test(trimmed)) {
    return {
      isValid: false,
      error: 'Project name must start with a letter or number',
    }
  }

  return { isValid: true }
}

/**
 * Description validation rules
 * MUST match backend: flowlib_server/app/models/projects.py:24
 */
export const validateDescription: Validator = (value: string): ValidationResult => {
  const trimmed = value.trim()

  if (trimmed.length > VALIDATION_LIMITS.DESCRIPTION_MAX_LENGTH) {
    return {
      isValid: false,
      error: `Description must not exceed ${VALIDATION_LIMITS.DESCRIPTION_MAX_LENGTH} characters`,
    }
  }

  return { isValid: true }
}

/**
 * Comma-separated list validation
 */
export const validateCommaSeparatedList: Validator = (value: string): ValidationResult => {
  const trimmed = value.trim()

  if (!trimmed) {
    // Empty list is valid
    return { isValid: true }
  }

  // Parse list
  const items = trimmed
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)

  // Check for duplicates
  const uniqueItems = new Set(items)
  if (uniqueItems.size !== items.length) {
    return {
      isValid: false,
      error: 'List contains duplicate items',
    }
  }

  // Validate each item
  for (const item of items) {
    if (item.length > VALIDATION_LIMITS.LIST_ITEM_MAX_LENGTH) {
      return {
        isValid: false,
        error: `Item "${item.slice(0, 20)}..." exceeds ${VALIDATION_LIMITS.LIST_ITEM_MAX_LENGTH} characters`,
      }
    }

    // Must be valid identifier-like string
    const validPattern = /^[a-zA-Z0-9_-]+$/
    if (!validPattern.test(item)) {
      return {
        isValid: false,
        error: `Item "${item}" contains invalid characters. Use only letters, numbers, hyphens, and underscores`,
      }
    }
  }

  return { isValid: true }
}

/**
 * Compose multiple validators
 */
export function composeValidators<T>(...validators: Validator<T>[]): Validator<T> {
  return (value: T): ValidationResult => {
    for (const validator of validators) {
      const result = validator(value)
      if (!result.isValid) {
        return result
      }
    }
    return { isValid: true }
  }
}

/**
 * Create a required field validator
 */
export function required(fieldName: string): Validator {
  return (value: string): ValidationResult => {
    if (!value || !value.trim()) {
      return {
        isValid: false,
        error: `${fieldName} is required`,
      }
    }
    return { isValid: true }
  }
}

/**
 * Create a min length validator
 */
export function minLength(min: number, fieldName: string): Validator {
  return (value: string): ValidationResult => {
    if (value.trim().length < min) {
      return {
        isValid: false,
        error: `${fieldName} must be at least ${min} characters`,
      }
    }
    return { isValid: true }
  }
}

/**
 * Create a max length validator
 */
export function maxLength(max: number, fieldName: string): Validator {
  return (value: string): ValidationResult => {
    if (value.trim().length > max) {
      return {
        isValid: false,
        error: `${fieldName} must not exceed ${max} characters`,
      }
    }
    return { isValid: true }
  }
}
