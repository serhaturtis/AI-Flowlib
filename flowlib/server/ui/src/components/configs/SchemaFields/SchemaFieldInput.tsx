import { SchemaField } from '../../../services/configs'
import { parseJson, stringifyJson } from '../../../utils/configs/configHelpers'
import { Button } from '../../ui/Button'
import { Input } from '../../ui/Input'
import { Textarea } from '../../ui/Textarea'
import { Select } from '../../ui/Select'
import { Label } from '../../ui/Label'

interface SchemaFieldInputProps {
  keyPrefix: string
  field: SchemaField
  value: unknown
  onChange: (val: unknown) => void
}


/**
 * Recursive component for rendering schema-driven form fields.
 * Supports: string, number, integer, boolean, object, array, union types.
 *
 * Key features:
 * - Type inference for union fields
 * - Nested object/array support
 * - Validation constraints (min/max, pattern, etc.)
 * - Enum support for dropdown selects
 */
export function SchemaFieldInput({
  keyPrefix,
  field,
  value,
  onChange,
}: SchemaFieldInputProps): JSX.Element {
  // Union support: when allowed_types is present, render a type selector and dispatch to the proper renderer
  if (Array.isArray(field.allowed_types) && field.allowed_types.length > 0) {
    const inferType = (val: unknown): string => {
      if (val === null || val === undefined || val === '') return field.allowed_types![0]
      if (typeof val === 'string') return 'string'
      if (typeof val === 'number') return Number.isInteger(val) ? 'integer' : 'number'
      if (typeof val === 'boolean') return 'boolean'
      if (Array.isArray(val)) return 'array'
      if (typeof val === 'object') return 'object'
      return field.allowed_types![0]
    }

    const coerceEmptyForType = (t: string): unknown => {
      if (t === 'string') return ''
      if (t === 'integer' || t === 'number') return ''
      if (t === 'boolean') return false
      if (t === 'array') return '[]'
      if (t === 'object') return '{}'
      return ''
    }

    const currentType = inferType(value)
    const labelText = `${field.name}${field.required ? ' *' : ''}`
    const description = field.description ? (
      <p className="text-sm text-muted-foreground">{field.description}</p>
    ) : null

    return (
      <fieldset
        key={`${keyPrefix}-${field.name}-union`}
        className="border border-border rounded-md p-3"
      >
        <legend className="px-2 text-sm font-medium">{labelText}</legend>
        {description}
        <div className="flex gap-2 items-center mb-2">
          <Label htmlFor={`${keyPrefix}-${field.name}-type-selector`}>Type</Label>
          <Select
            id={`${keyPrefix}-${field.name}-type-selector`}
            value={currentType}
            onChange={(e) => {
              const nextType = e.target.value
              onChange(coerceEmptyForType(nextType))
            }}
          >
            {field.allowed_types!.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </Select>
        </div>
        <SchemaFieldInput
          keyPrefix={`${keyPrefix}-${field.name}-union-inner`}
          field={{ ...field, type: currentType, allowed_types: undefined }}
          value={value}
          onChange={onChange}
        />
      </fieldset>
    )
  }

  const labelText = `${field.name}${field.required ? ' *' : ''}`
  const description = field.description ? (
    <p className="text-sm text-muted-foreground">{field.description}</p>
  ) : null

  // String field
  if (field.type === 'string') {
    if (field.enum && field.enum.length > 0) {
      const stringValue = typeof value === 'string' ? value : value ?? ''
      return (
        <div key={`${keyPrefix}-${field.name}`} className="grid gap-2">
          <Label htmlFor={`${keyPrefix}-${field.name}-input`}>{labelText}</Label>
          <Select
            id={`${keyPrefix}-${field.name}-input`}
            value={stringValue as string}
            onChange={(event) => onChange(event.target.value)}
          >
            {!field.required && <option value="">—</option>}
            {field.enum.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </Select>
          {description}
        </div>
      )
    }

    const stringValue = typeof value === 'string' ? value : value ?? ''
    return (
      <div key={`${keyPrefix}-${field.name}`} className="grid gap-2">
        <Label htmlFor={`${keyPrefix}-${field.name}-input`}>{labelText}</Label>
        <Input
          id={`${keyPrefix}-${field.name}-input`}
          type="text"
          value={stringValue as string}
          onChange={(event) => onChange(event.target.value)}
          minLength={field.string_min_length ?? undefined}
          maxLength={field.string_max_length ?? undefined}
          pattern={field.pattern ?? undefined}
        />
        {description}
      </div>
    )
  }

  // Boolean field
  if (field.type === 'boolean') {
    const boolValue =
      typeof value === 'boolean' ? value : typeof value === 'string' ? value === 'true' : Boolean(value)
    return (
      <div key={`${keyPrefix}-${field.name}`} className="grid gap-2">
        <Label htmlFor={`${keyPrefix}-${field.name}-input`}>{labelText}</Label>
        <Select
          id={`${keyPrefix}-${field.name}-input`}
          value={boolValue ? 'true' : 'false'}
          onChange={(event) => onChange(event.target.value === 'true')}
        >
          <option value="true">true</option>
          <option value="false">false</option>
        </Select>
        {description}
      </div>
    )
  }

  // Number field (integer or number)
  if (field.type === 'integer' || field.type === 'number') {
    const stringValue =
      typeof value === 'number' || typeof value === 'string' ? String(value ?? '') : value ?? ''
    return (
      <div key={`${keyPrefix}-${field.name}`} className="grid gap-2">
        <Label htmlFor={`${keyPrefix}-${field.name}-input`}>{labelText}</Label>
        <Input
          id={`${keyPrefix}-${field.name}-input`}
          type="number"
          value={stringValue as string}
          onChange={(event) => onChange(event.target.value)}
          min={field.numeric_min ?? undefined}
          max={field.numeric_max ?? undefined}
        />
        {description}
      </div>
    )
  }

  // Object field with nested children
  if (field.type === 'object' && field.children && field.children.length > 0) {
    let currentObj: Record<string, unknown>
    if (typeof value === 'string') {
      const parseResult = parseJson<Record<string, unknown>>(value)
      currentObj = parseResult.success && parseResult.data ? parseResult.data : {}
    } else {
      currentObj = (value as Record<string, unknown>) || {}
    }

    const handleChildChange = (childName: string, childVal: unknown) => {
      const next = { ...(currentObj || {}) }
      next[childName] = childVal
      onChange(JSON.stringify(next, null, 2))
    }

    return (
      <fieldset
        key={`${keyPrefix}-${field.name}`}
        className="border border-border rounded-md p-3"
      >
        <legend className="px-2 text-sm font-medium">{labelText}</legend>
        {description}
        <div className="grid gap-3">
          {field.children?.map((child) => (
            <SchemaFieldInput
              key={`${keyPrefix}-${field.name}-${child.name}`}
              keyPrefix={`${keyPrefix}-${field.name}`}
              field={child}
              value={currentObj?.[child.name]}
              onChange={(v) => handleChildChange(child.name, v)}
            />
          ))}
        </div>
      </fieldset>
    )
  }

  // Array field
  if (field.type === 'array') {
    // Array of objects with children
    if (field.children && field.children.length > 0) {
      const currentArr =
        typeof value === 'string'
          ? (() => {
              try {
                const parsed = JSON.parse(value as string)
                return Array.isArray(parsed) ? parsed : []
              } catch {
                return []
              }
            })()
          : (Array.isArray(value) ? value : []) as Array<Record<string, unknown>>

      const handleItemChange = (index: number, childName: string, childVal: unknown) => {
        const next = [...currentArr]
        const item = { ...(next[index] || {}) }
        item[childName] = childVal
        next[index] = item
        onChange(JSON.stringify(next, null, 2))
      }

      const addItem = () => {
        const next = [...currentArr, {}]
        onChange(JSON.stringify(next, null, 2))
      }

      const removeItem = (index: number) => {
        const next = currentArr.filter((_, i) => i !== index)
        onChange(JSON.stringify(next, null, 2))
      }

      return (
        <fieldset
          key={`${keyPrefix}-${field.name}`}
          className="border border-border rounded-md p-3"
        >
          <legend className="px-2 text-sm font-medium">{labelText}</legend>
          {description}
          <div className="grid gap-3">
            {currentArr.map((item, idx) => (
              <fieldset
                key={`${keyPrefix}-${field.name}-item-${idx}`}
                className="border border-dashed border-border rounded-md p-3"
              >
                <legend className="px-2 text-sm font-medium">Item {idx + 1}</legend>
                <div className="grid gap-2">
                  {field.children?.map((child) => (
                    <SchemaFieldInput
                      key={`${keyPrefix}-${field.name}-item-${idx}-${child.name}`}
                      keyPrefix={`${keyPrefix}-${field.name}-item-${idx}`}
                      field={child}
                      value={(item as Record<string, unknown>)?.[child.name]}
                      onChange={(v) => handleItemChange(idx, child.name, v)}
                    />
                  ))}
                </div>
                <div className="mt-2">
                  <Button type="button" onClick={() => removeItem(idx)} variant="destructive" size="sm">
                    Remove
                  </Button>
                </div>
              </fieldset>
            ))}
            <div>
              <Button type="button" onClick={addItem} variant="outline" size="sm">
                Add Item
              </Button>
            </div>
          </div>
        </fieldset>
      )
    }

    // Primitive arrays: newline separated; bridge to JSON array
    const allowedItemTypes = field.items_allowed_types && field.items_allowed_types.length > 0 ? field.items_allowed_types : null

    const currentListStrings: string[] =
      typeof value === 'string'
        ? (() => {
            try {
              const parsed = JSON.parse(value as string)
              if (Array.isArray(parsed)) return parsed.map((v) => String(v ?? '')).filter(Boolean)
              return []
            } catch {
              return (value as string).split('\n').map((s) => s.trim()).filter(Boolean)
            }
          })()
        : Array.isArray(value)
          ? (value as unknown[]).map((v) => String(v ?? '')).filter(Boolean)
          : []

    const guessItemTypeFromParsed = (): string => {
      // Try to infer from first parsed item if JSON was provided
      if (typeof value === 'string') {
        try {
          const parsed = JSON.parse(value as string)
          if (Array.isArray(parsed) && parsed.length > 0) {
            const v = parsed[0]
            if (typeof v === 'number') return Number.isInteger(v) ? 'integer' : 'number'
            if (typeof v === 'boolean') return 'boolean'
          }
        } catch {
          // Ignore parse errors
        }
      }
      return field.items_type || 'string'
    }

    const activeItemType = allowedItemTypes
      ? (allowedItemTypes.includes(guessItemTypeFromParsed()) ? guessItemTypeFromParsed() : allowedItemTypes[0])
      : (field.items_type || 'string')

    const coerceItem = (raw: string, tOverride?: string): unknown => {
      const t = tOverride || activeItemType
      if (t === 'integer') {
        const n = Number(raw)
        return Number.isFinite(n) ? Math.trunc(n) : raw
      }
      if (t === 'number') {
        const n = Number(raw)
        return Number.isFinite(n) ? n : raw
      }
      if (t === 'boolean') {
        if (raw.toLowerCase() === 'true') return true
        if (raw.toLowerCase() === 'false') return false
        return raw
      }
      return raw
    }

    const onListChange = (text: string) => {
      const arr = text.split('\n').map((s) => s.trim()).filter(Boolean).map((s) => coerceItem(s))
      onChange(JSON.stringify(arr, null, 2))
    }

    return (
      <div key={`${keyPrefix}-${field.name}`} className="grid gap-2">
        <Label htmlFor={`${keyPrefix}-${field.name}-textarea`}>{labelText}</Label>
        {allowedItemTypes && (
          <div className="flex gap-2 items-center">
            <Label htmlFor={`${keyPrefix}-${field.name}-item-type`} className="text-xs text-muted-foreground">
              Item type
            </Label>
            <Select
              id={`${keyPrefix}-${field.name}-item-type`}
              value={activeItemType}
              onChange={(e) => {
                const t = e.target.value
                const coerced = currentListStrings.map((s) => coerceItem(s, t))
                onChange(JSON.stringify(coerced, null, 2))
              }}
            >
              {allowedItemTypes.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </Select>
          </div>
        )}
        <Textarea
          id={`${keyPrefix}-${field.name}-textarea`}
          rows={4}
          value={currentListStrings.join('\n')}
          onChange={(e) => onListChange(e.target.value)}
          placeholder="One value per line"
          className="font-mono"
        />
        {description}
      </div>
    )
  }

  // Fallback: render as JSON textarea
  let jsonValue: string
  if (typeof value === 'string') {
    jsonValue = value
  } else if (value === undefined || value === null) {
    jsonValue = ''
  } else {
    const result = stringifyJson(value)
    jsonValue = result.data ?? ''
  }

  return (
    <div key={`${keyPrefix}-${field.name}`} className="grid gap-2">
      <Label htmlFor={`${keyPrefix}-${field.name}-textarea`}>{labelText}</Label>
      <Textarea
        id={`${keyPrefix}-${field.name}-textarea`}
        rows={6}
        value={jsonValue}
        onChange={(event) => onChange(event.target.value)}
        className="font-mono"
      />
      {description}
    </div>
  )
}
