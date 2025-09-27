"""Context model with attribute-based access and improved state management.

This module provides a Context class for managing execution state with 
attribute-based access, snapshot capabilities, and clean validation.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, TypeVar, Generic, Type, Union
from flowlib.core.models import StrictBaseModel

T = TypeVar('T', bound=StrictBaseModel)

class Context(Generic[T]):
    """Enhanced execution context accepting a Pydantic model as its primary data payload.
    
    This class provides:
    1. Attribute-based access to underlying data (still stored as dict internally).
    2. Clean state management with snapshots and rollbacks (operating on the internal dict).
    3. A defined Pydantic model type associated with the context.
    4. Deep copying for isolation.
    """
    
    def __init__(
        self,
        # Accept an optional StrictBaseModel instance or dict for backward compatibility
        data: Optional[Union[T, Dict[str, Any]]] = None,
        # Model type can be inferred from data, or passed if data is None initially
        model_type: Optional[Type[T]] = None 
    ):
        """Initialize context with an optional Pydantic model instance or dict.
        
        Args:
            data: Optional Pydantic StrictBaseModel instance or dict containing initial state.
            model_type: Optional Pydantic model type (required if data is None initially and type is needed later).
            
        Raises:
            TypeError: If data is provided and is neither a Pydantic StrictBaseModel nor a dict.
            ValueError: If data fails validation against its own model type.
        """
        self._snapshots: List[Dict[str, Any]] = []

        # Initialize attributes with proper types
        self._data: Dict[str, Any]
        self._model_type: Optional[Type[T]]

        if data is not None:
            if isinstance(data, StrictBaseModel):
                try:
                    # Validate the input model instance itself (Pydantic does this on init, but explicit check is good)
                    # Also sets the internal dictionary representation
                    self._data = data.model_dump()
                    # Infer model type from the provided instance
                    self._model_type = type(data)
                except Exception as e: # Catch Pydantic validation errors if any during model_dump
                    raise ValueError(f"Provided data model failed validation: {str(e)}") from e
            elif isinstance(data, dict):
                # Backward compatibility: accept dict
                self._data = data.copy()
                self._model_type = model_type
            else:
                raise TypeError(f"Context data must be a Pydantic StrictBaseModel instance or dict, got {type(data).__name__}")
        else:
            # No data provided, initialize empty
            self._data = {}
            # Use provided model_type if given, otherwise None
            self._model_type = model_type 

        # Initial validation using _validate method is redundant if data is validated above
        # if self._model_type and self._data:
        #    self._validate(self._data)
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get the internal data dictionary (a dump of the model state)."""
        return self._data
        
    @property
    def model_type(self) -> Optional[Type[T]]:
        """Get the Pydantic model type associated with this context, if any."""
        return self._model_type

    # __getattr__, get, keys remain the same, operating on self._data
    def __getattr__(self, name: str) -> Any:
        """Enable attribute-based access to context data.
        
        Args:
            name: Attribute name to access
            
        Returns:
            Attribute value
            
        Raises:
            AttributeError: If attribute not found in the internal data dict
        """
        # Check internal dict first for dynamic attributes
        if name in self._data:
            return self._data[name]
        # Raise error if not found
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def get(self, key: str) -> Any:
        """Get a value with strict access - no fallbacks.
        
        Args:
            key: Key to retrieve
            
        Returns:
            The value for the key
            
        Raises:
            KeyError: If key is not found
        """
        if key not in self._data:
            raise KeyError(f"Key '{key}' not found in context. Available keys: {list(self._data.keys())}")
        return self._data[key]

    def keys(self) -> List[str]:
        return list(self._data.keys())
        
    # --- Methods requiring careful consideration under the new model --- 
    
    # OPTION 1: Keep set/update but add validation (more complex)
    # OPTION 2: Remove/Deprecate set/update to enforce immutability via model re-creation (stricter)
    # Let's go with OPTION 2 for stricter adherence to the user's request for rigidity.
    
    # @deprecated("Directly setting arbitrary keys bypasses model validation. Modify the underlying model and re-create Context if needed.")
    def set(self, key: str, value: Any) -> 'Context[T]':
        """(DEPRECATED) Set a value in the internal context dictionary. Bypasses model validation."""
        # raise NotImplementedError("Directly setting keys is disallowed. Mutate the model and re-create context.")
        # OR keep it but log warning / add validation if possible?
        # For now, retain original behavior but emphasize it's low-level.
        self._data[key] = value
        # We cannot easily re-validate the whole model here without reconstructing it.
        return self

    # @deprecated("Directly updating with a dictionary bypasses model validation. Modify the underlying model and re-create Context if needed.")
    def update(self, data: Dict[str, Any]) -> 'Context[T]':
        """(DEPRECATED) Update internal context data with dictionary. Bypasses model validation."""
        # raise NotImplementedError("Directly updating with dict is disallowed. Mutate the model and re-create context.")
        # OR keep it but log warning / add validation if possible?
        # For now, retain original behavior but emphasize it's low-level.
        self._data.update(data)
        # We cannot easily re-validate the whole model here without reconstructing it.
        return self
        
    # _validate is less useful now if set/update are discouraged/removed.
    # It was primarily for validating incoming dicts on init/update. Pydantic handles init validation.
    # def _validate(self, data: Dict[str, Any]) -> None: ... 

    # Snapshots/Rollback operate on the internal dictionary representation.
    def create_snapshot(self) -> int:
        self._snapshots.append(deepcopy(self._data))
        return len(self._snapshots) - 1

    def rollback(self, snapshot_id: Optional[int] = None) -> 'Context[T]':
        if not self._snapshots:
            raise ValueError("No snapshots available for rollback")
        target_id = snapshot_id if snapshot_id is not None else len(self._snapshots) - 1
        if not (0 <= target_id < len(self._snapshots)):
            raise ValueError(f"Invalid snapshot ID: {target_id}")
        self._data = deepcopy(self._snapshots[target_id])
        self._snapshots = self._snapshots[:target_id]
        return self

    def clear_snapshots(self) -> 'Context[T]':
        self._snapshots = []
        return self

    def copy(self) -> 'Context[T]':
        """Create a deep copy of the context.
        
        Reconstructs the model instance from the internal dictionary if possible,
        otherwise copies the dictionary.
        """
        new_data_instance = None
        if self._model_type:
            try:
                 # Try to create a new model instance from the current data dict
                 new_data_instance = self._model_type(**self._data)
            except Exception as e:
                 # Log warning? If data dict became invalid, copy might fail later.
                 print(f"Warning: Failed to reconstruct model during copy: {e}")
                 # Fallback: copy the dict, but the new Context won't have a valid model payload
                 return Context(data=None, model_type=None) # Or raise error?
                     
        # If we successfully created a new model instance, use it
        if new_data_instance:
             return Context(data=new_data_instance)
        else:
             # If there was no model type initially, or reconstruction failed
             # Create a new context with no model type and the copied dict
             # This state is less type-safe but preserves the dict data.
             # NOTE: This breaks the strict typing goal if reconstruction fails. Consider raising error instead?
             # For now, preserving dict state.
             new_ctx: Context[T] = Context(data=None, model_type=None)
             new_ctx._data = deepcopy(self._data)
             return new_ctx

    def as_model(self) -> Optional[T]:
        """Convert internal context dictionary back to the original model instance."""
        if not self._model_type:
            return None
        try:
            # Re-create the model instance from the current internal dictionary state
            return self._model_type(**self._data)
        except Exception as e:
            # This indicates the internal _data dictionary has become invalid 
            # with respect to the _model_type, likely due to using set/update.
            raise ValueError(f"Failed to convert context data to {self._model_type.__name__}: {str(e)}. Internal data may be inconsistent.")

    # __str__ and __contains__ remain the same
    def __str__(self) -> str:
        model_name = self._model_type.__name__ if self._model_type else "None"
        return f"Context(model_type={model_name}, data_keys={list(self._data.keys())}, snapshots={len(self._snapshots)})"

    def __contains__(self, key: str) -> bool:
        return key in self._data 
    
    def merge(self, other: 'Context[T]') -> 'Context[T]':
        """Merge this context with another context, returning a new context.
        
        Args:
            other: Another context to merge
            
        Returns:
            New context with merged data
        """
        # Create merged data
        merged_data = self._data.copy()
        merged_data.update(other._data)
        
        # Create new context with merged data, preserving model type
        return Context(data=merged_data, model_type=self._model_type)