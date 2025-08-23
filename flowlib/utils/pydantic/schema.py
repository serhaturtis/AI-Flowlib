"""
Pydantic schema utilities.

This module provides utilities for working with Pydantic models and their JSON schemas,
particularly handling recursive conversion and formatting of nested models.
"""

from typing import Any, Dict, List, Optional, Type, Union, get_type_hints, get_origin, get_args
from pydantic import BaseModel
import inspect
import json
import copy


def model_to_schema_dict(model: Type[BaseModel]) -> Dict[str, Any]:
    """
    Convert a Pydantic model class to a JSON schema dictionary.
    
    This function handles nested Pydantic models recursively, ensuring that
    the complete schema is properly generated.
    
    Args:
        model: A Pydantic model class (not an instance)
        
    Returns:
        A dictionary representation of the JSON schema
        
    Raises:
        TypeError: If the input is not a Pydantic model class
    """
    if not inspect.isclass(model) or not issubclass(model, BaseModel):
        raise TypeError(f"Expected a Pydantic model class, got {type(model).__name__}")
    
    # Get the basic schema from model_json_schema
    try:
        schema = model.model_json_schema()
    except AttributeError:
        # Fall back to older schema_json method for backward compatibility
        if hasattr(model, "schema") and callable(getattr(model, "schema")):
            schema = model.schema()
        else:
            raise TypeError(f"Model {model.__name__} doesn't support schema generation")
    
    # Process nested models if this is a complex schema
    if "properties" in schema:
        _process_nested_models(schema, model)
    
    return schema


def _process_nested_models(schema: Dict[str, Any], model: Type[BaseModel]) -> None:
    """
    Process properties in a schema to handle nested Pydantic models.
    
    This function modifies the schema in place, adding detailed schema information
    for any nested Pydantic models found in the properties.
    
    Args:
        schema: The schema dictionary to process
        model: The Pydantic model class that this schema represents
    """
    # Get type hints for all fields
    try:
        type_hints = get_type_hints(model)
    except Exception:
        # If we can't get type hints, we can't process nested models
        return
    
    # Process each property
    properties = schema["properties"] if "properties" in schema else {}
    for prop_name, prop_info in properties.items():
        if prop_name in type_hints:
            field_type = type_hints[prop_name]
            
            # Handle Optional types (Union[Type, None])
            origin = get_origin(field_type)
            if origin is Union:
                args = get_args(field_type)
                # If it's an Optional (Union with NoneType)
                if len(args) == 2 and type(None) in args:
                    # Get the non-None type
                    field_type = next(arg for arg in args if arg is not type(None))
            
            # Check if this is a Pydantic model
            if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
                # Replace with the full schema for this nested model
                nested_schema = model_to_schema_dict(field_type)
                prop_info.update(nested_schema)
            
            # Handle List[PydanticModel]
            elif origin is list or origin is List:
                args = get_args(field_type)
                if args and inspect.isclass(args[0]) and issubclass(args[0], BaseModel):
                    if "items" in prop_info:
                        nested_schema = model_to_schema_dict(args[0])
                        prop_info["items"].update(nested_schema)
            
            # Handle Dict[str, PydanticModel]
            elif origin is dict or origin is Dict:
                args = get_args(field_type)
                if (len(args) == 2 and args[0] is str and 
                    inspect.isclass(args[1]) and issubclass(args[1], BaseModel)):
                    if "additionalProperties" in prop_info:
                        nested_schema = model_to_schema_dict(args[1])
                        prop_info["additionalProperties"].update(nested_schema)


def model_to_json_schema(model: Type[BaseModel], indent: int = 2) -> str:
    """
    Convert a Pydantic model to a formatted JSON schema string.
    
    This function processes the model recursively, including any nested models,
    and returns a properly indented JSON string representation.
    
    Args:
        model: A Pydantic model class (not an instance)
        indent: Number of spaces for indentation in the JSON output
        
    Returns:
        Formatted JSON schema as a string
        
    Raises:
        TypeError: If the input is not a Pydantic model class
    """
    schema_dict = model_to_schema_dict(model)
    return json.dumps(schema_dict, indent=indent, sort_keys=False, ensure_ascii=False)


def clean_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean up a schema dictionary by removing redundancy and simplifying the output.
    
    Args:
        schema: The schema dictionary to clean
        
    Returns:
        A cleaned schema dictionary
    """
    result = copy.deepcopy(schema)
    
    # Remove redundant descriptions (those contained in property definitions)
    if "$defs" in result:
        for def_key, def_value in result["$defs"].items():
            # Simplify description by removing attribute lists
            if "description" in def_value and "\n\nAttributes:" in def_value["description"]:
                def_value["description"] = def_value["description"].split("\n\nAttributes:")[0]
            
            # Process properties of the definition
            if "properties" in def_value:
                for prop_key, prop_value in def_value["properties"].items():
                    # Simplify property descriptions
                    if "description" in prop_value:
                        prop_value["description"] = prop_value["description"].split("\n")[0]

    # Process properties in the main schema
    if "properties" in result:
        for prop_key, prop_value in result["properties"].items():
            # Simplify property descriptions
            if "description" in prop_value:
                prop_value["description"] = prop_value["description"].split("\n")[0]
            
            # If property has a $ref, remove redundant fields that duplicate info from the referenced definition
            if "$ref" in prop_value:
                ref_key = prop_value["$ref"].split("/")[-1]
                # Keep only $ref, title, and description
                keys_to_keep = ["$ref", "title", "description"]
                for key in list(prop_value.keys()):
                    if key not in keys_to_keep:
                        del prop_value[key]
    
    # Simplify main schema description
    if "description" in result and "\n\nAttributes:" in result["description"]:
        result["description"] = result["description"].split("\n\nAttributes:")[0]
    
    return result


def model_to_clean_json_schema(model: Type[BaseModel], indent: int = 2) -> str:
    """
    Convert a Pydantic model to a clean, simplified JSON schema string.
    
    Args:
        model: A Pydantic model class
        indent: Number of spaces for indentation (default: 2)
        
    Returns:
        A simplified JSON schema string
    """
    schema_dict = model_to_schema_dict(model)
    clean_dict = clean_schema(schema_dict)
    return json.dumps(clean_dict, indent=indent)


def format_schema_for_prompt(schema: Dict[str, Any], include_descriptions: bool = True) -> str:
    """
    Format a JSON schema as a human-readable string suitable for a prompt.
    
    Args:
        schema: JSON schema dictionary
        include_descriptions: Whether to include field descriptions
        
    Returns:
        A formatted string representation of the schema
    """
    title = schema["title"] if "title" in schema else "Object"
    output = f"Schema: {title}\n"
    
    # Add type information
    schema_type = schema["type"] if "type" in schema else "object"
    output += f"Type: {schema_type}\n"
    
    # Add description if available
    if "description" in schema and schema["description"]:
        output += f"Description: {schema['description']}\n"
    
    # Properties section
    if "properties" in schema:
        output += "\nFields:\n"
        for field_name, field_info in schema["properties"].items():
            field_type = field_info["type"] if "type" in field_info else "unknown"
            
            # Handle arrays with items
            if field_type == "array" and "items" in field_info:
                items_type = field_info["items"]["type"] if "type" in field_info["items"] else "unknown"
                if "title" in field_info["items"]:
                    items_type = field_info["items"]["title"]
                field_type = f"array of {items_type}"
            
            # Handle objects with a title
            if field_type == "object" and "title" in field_info:
                field_type = field_info["title"]
            
            # Add field to output
            output += f"- {field_name} ({field_type})"
            
            # Add description if available and requested
            if include_descriptions and "description" in field_info and field_info["description"]:
                output += f": {field_info['description']}"
            
            output += "\n"
    
    # Required fields
    if "required" in schema and schema["required"]:
        required_fields = ", ".join(schema["required"])
        output += f"\nRequired fields: {required_fields}\n"
    
    return output


def get_model_schema_text(model: Type[BaseModel], include_descriptions: bool = True) -> str:
    """
    Get a human-readable text representation of a Pydantic model's schema.
    
    This is a convenience function that combines model_to_schema_dict and
    format_schema_for_prompt.
    
    Args:
        model: A Pydantic model class
        include_descriptions: Whether to include field descriptions
        
    Returns:
        A formatted string representation of the model's schema
        
    Raises:
        TypeError: If the input is not a Pydantic model class
    """
    schema = model_to_schema_dict(model)
    return format_schema_for_prompt(schema, include_descriptions)


def save_model_schema_to_file(model: Type[BaseModel], file_path: str, indent: int = 2, clean: bool = True) -> None:
    """
    Save a Pydantic model's schema to a JSON file.
    
    Args:
        model: A Pydantic model class
        file_path: Path where the JSON file should be saved
        indent: Number of spaces for indentation in the JSON output
        clean: Whether to clean the schema before saving
        
    Raises:
        TypeError: If the input is not a Pydantic model class
        IOError: If there is an error writing to the file
    """
    if clean:
        schema_json = model_to_clean_json_schema(model, indent)
    else:
        schema_json = model_to_json_schema(model, indent)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(schema_json)
    except IOError as e:
        raise IOError(f"Failed to write schema to file {file_path}: {str(e)}")


def get_model_example(model: Type[BaseModel]) -> Dict[str, Any]:
    """
    Generate an example instance of a Pydantic model with default values.
    
    This function creates an example dictionary that could be used to instantiate
    the model, using default values where available.
    
    Args:
        model: A Pydantic model class
        
    Returns:
        Dictionary with example values for the model fields
        
    Raises:
        TypeError: If the input is not a Pydantic model class
    """
    if not inspect.isclass(model) or not issubclass(model, BaseModel):
        raise TypeError(f"Expected a Pydantic model class, got {type(model).__name__}")
    
    schema = model_to_schema_dict(model)
    example = {}
    
    properties = schema["properties"] if "properties" in schema else {}
    for field_name, field_info in properties.items():
        # Try to use example if provided in schema
        if "example" in field_info:
            example[field_name] = field_info["example"]
            continue
            
        # Otherwise use a type-appropriate default
        field_type = field_info["type"] if "type" in field_info else "unknown"
        
        if field_type == "string":
            example[field_name] = f"example_{field_name}"
        elif field_type == "integer":
            example[field_name] = 0
        elif field_type == "number":
            example[field_name] = 0.0
        elif field_type == "boolean":
            example[field_name] = False
        elif field_type == "array":
            example[field_name] = []
        elif field_type == "object":
            example[field_name] = {}
        else:
            example[field_name] = None
    
    return example


def simplify_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create an ultra-simplified schema with minimal information for easy reading.
    
    This produces a very compact schema focused only on essential information:
    - Field names
    - Types
    - Short descriptions (one line)
    - Required fields
    
    Args:
        schema: The schema dictionary to simplify
        
    Returns:
        A highly simplified schema dictionary with flat structure (no $defs)
    """
    result = {
        "title": schema["title"] if "title" in schema else "",
        "description": (schema["description"] if "description" in schema else "").split("\n")[0]
    }
    
    # Create definitions lookup table to resolve references
    definitions = {}
    if "$defs" in schema:
        for def_name, def_info in schema["$defs"].items():
            definitions[def_name] = def_info
    
    # Process properties
    if "properties" in schema:
        result["properties"] = {}
        for prop_name, prop_info in schema["properties"].items():
            # Get type information
            prop_type = ""
            if "$ref" in prop_info:
                # Get referenced type instead of using $ref
                ref_name = prop_info["$ref"].split("/")[-1]
                prop_type = ref_name
                
                # If the property has a title, use it as type name
                if "title" in prop_info:
                    prop_type = prop_info["title"]
                # Otherwise use referenced definition's title if available
                elif ref_name in definitions and "title" in definitions[ref_name]:
                    prop_type = definitions[ref_name]["title"]
            elif "type" in prop_info:
                prop_type = prop_info["type"]
                # Add item type for arrays
                if prop_type == "array" and "items" in prop_info:
                    if "type" in prop_info["items"]:
                        prop_type = f"array[{prop_info['items']['type']}]"
                    elif "$ref" in prop_info["items"]:
                        ref_name = prop_info["items"]["$ref"].split("/")[-1]
                        prop_type = f"array[{ref_name}]"
            
            # Get description (first line only)
            description = ""
            if "description" in prop_info:
                description = prop_info["description"].split("\n")[0]
            elif "$ref" in prop_info:
                # If property has a $ref, use the referenced definition's description
                ref_name = prop_info["$ref"].split("/")[-1]
                if ref_name in definitions and "description" in definitions[ref_name]:
                    description = definitions[ref_name]["description"].split("\n")[0]
            
            # Create simplified property info
            result["properties"][prop_name] = f"{prop_type}: {description}"
    
    # Add required fields
    if "required" in schema:
        result["required"] = schema["required"]
    
    # Instead of using definitions, add sub-schemas as separate top-level entries
    if "$defs" in schema:
        for def_name, def_info in schema["$defs"].items():
            sub_schema = {
                "title": def_info["title"] if "title" in def_info else def_name,
                "description": (def_info["description"] if "description" in def_info else "").split("\n")[0],
                "properties": {}
            }
            
            # Process properties of the sub-schema
            if "properties" in def_info:
                for prop_name, prop_info in def_info["properties"].items():
                    # Get type
                    prop_type = prop_info["type"] if "type" in prop_info else ""
                    if "anyOf" in prop_info:
                        types = []
                        for type_info in prop_info["anyOf"]:
                            if "type" in type_info:
                                types.append(type_info["type"])
                        if types:
                            prop_type = " | ".join(types)
                    
                    # Get description
                    description = ""
                    if "description" in prop_info:
                        description = prop_info["description"].split("\n")[0]
                    
                    # Create simplified property
                    sub_schema["properties"][prop_name] = f"{prop_type}: {description}"
            
            # Add required fields
            if "required" in def_info:
                sub_schema["required"] = def_info["required"]
                
            # Add this schema with a meaningful name (no $defs)
            result[def_name] = sub_schema
    
    return result


def model_to_simple_json_schema(model: Type[BaseModel], indent: int = 2) -> str:
    """
    Convert a Pydantic model to an ultra-simplified JSON schema string.
    
    This format is designed for maximum readability with minimal technical details
    and no $defs structure - everything is flattened to the top level.
    
    Args:
        model: A Pydantic model class
        indent: Number of spaces for indentation (default: 2)
        
    Returns:
        A highly simplified JSON schema string with flat structure
    """
    schema_dict = model_to_schema_dict(model)
    simple_dict = simplify_schema(schema_dict)
    return json.dumps(simple_dict, indent=indent) 