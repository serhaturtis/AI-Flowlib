"""Glob tool - opencode style implementation with flowlib conventions."""

import os
import glob
import fnmatch
from typing import Dict, Any, List
from .base import REPLTool, ToolResult, ToolExecutionContext


class GlobTool(REPLTool):
    """Find files matching glob patterns with modification time sorting."""
    
    async def init(self) -> Dict[str, Any]:
        """Initialize glob tool schema."""
        return {
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match files against (e.g., '**/*.py', 'src/**/*.ts')"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: current directory)",
                        "default": "."
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Include hidden files and directories",
                        "default": False
                    },
                    "include_dirs": {
                        "type": "boolean", 
                        "description": "Include directories in results",
                        "default": False
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["name", "mtime", "size"],
                        "description": "Sort results by name, modification time, or size",
                        "default": "mtime"
                    },
                    "reverse": {
                        "type": "boolean",
                        "description": "Reverse sort order",
                        "default": False
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return", 
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 100
                    }
                },
                "required": ["pattern"],
                "additionalProperties": False
            }
        }
    
    async def execute(self, parameters: Dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        """Execute glob search."""
        pattern = parameters["pattern"]
        path = parameters.get("path", ".")
        include_hidden = parameters.get("include_hidden", False)
        include_dirs = parameters.get("include_dirs", False)
        sort_by = parameters.get("sort_by", "mtime")
        reverse = parameters.get("reverse", False)
        max_results = parameters.get("max_results", 100)
        
        try:
            # Resolve path
            if not os.path.isabs(path):
                path = os.path.join(context.working_directory, path)
            path = os.path.abspath(path)
            
            # Validate path is within working directory
            working_dir = os.path.abspath(context.working_directory)
            if not path.startswith(working_dir):
                return ToolResult.error(f"Path {path} is not in the current working directory")
            
            if not os.path.exists(path):
                return ToolResult.error(f"Directory does not exist: {path}")
            
            if not os.path.isdir(path):
                return ToolResult.error(f"Path is not a directory: {path}")
            
            # Change to search directory for glob
            original_cwd = os.getcwd()
            os.chdir(path)
            
            try:
                # Find matches using glob
                matches = glob.glob(pattern, recursive=True)
                
                # Filter results
                filtered_matches = []
                for match in matches:
                    abs_match = os.path.abspath(match)
                    
                    # Skip hidden files/dirs if not included
                    if not include_hidden:
                        path_parts = match.split(os.sep)
                        if any(part.startswith('.') for part in path_parts):
                            continue
                    
                    # Check if it's a directory
                    is_dir = os.path.isdir(abs_match)
                    
                    # Include based on type filter
                    if is_dir and not include_dirs:
                        continue
                    if not is_dir and not os.path.isfile(abs_match):
                        continue  # Skip special files
                    
                    # Add file info
                    try:
                        stat = os.stat(abs_match)
                        filtered_matches.append({
                            'path': match,
                            'abs_path': abs_match,
                            'is_dir': is_dir,
                            'size': stat.st_size,
                            'mtime': stat.st_mtime
                        })
                    except OSError:
                        continue  # Skip files we can't stat
                
                # Sort results
                if sort_by == "name":
                    filtered_matches.sort(key=lambda x: x['path'].lower())
                elif sort_by == "mtime":
                    filtered_matches.sort(key=lambda x: x['mtime'])
                elif sort_by == "size":
                    filtered_matches.sort(key=lambda x: x['size'])
                
                if reverse:
                    filtered_matches.reverse()
                
                # Limit results
                total_found = len(filtered_matches)
                if len(filtered_matches) > max_results:
                    filtered_matches = filtered_matches[:max_results]
                    truncated = True
                else:
                    truncated = False
                
                # Format output
                if not filtered_matches:
                    output = "No files found matching pattern"
                else:
                    output_lines = []
                    for match in filtered_matches:
                        line = match['path']
                        if include_dirs and match['is_dir']:
                            line += "/"
                        output_lines.append(line)
                    output = "\\n".join(output_lines)
                
                metadata = {
                    "pattern": pattern,
                    "search_path": path,
                    "total_found": total_found,
                    "returned": len(filtered_matches),
                    "truncated": truncated,
                    "sort_by": sort_by,
                    "reverse": reverse,
                    "include_hidden": include_hidden,
                    "include_dirs": include_dirs
                }
                
                return ToolResult.success(output, metadata)
                
            finally:
                os.chdir(original_cwd)
                
        except glob.GlobError as e:
            return ToolResult.error(f"Invalid glob pattern: {e}")
        except Exception as e:
            return ToolResult.error(f"Error during glob search: {str(e)}")