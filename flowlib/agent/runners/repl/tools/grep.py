"""Grep tool - opencode style implementation with flowlib conventions."""

import os
import re
import fnmatch
from typing import Dict, Any, List, Tuple, Optional
from .base import REPLTool, ToolResult, ToolExecutionContext


class GrepTool(REPLTool):
    """Search for patterns in file contents with advanced filtering."""
    
    async def init(self) -> Dict[str, Any]:
        """Initialize grep tool schema."""
        return {
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression pattern to search for"
                    },
                    "path": {
                        "type": "string", 
                        "description": "Directory or file to search in (default: current directory)",
                        "default": "."
                    },
                    "glob": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g., '*.py', '*.{js,ts}')"
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Case insensitive search",
                        "default": False
                    },
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count"],
                        "description": "Output mode: content shows matching lines, files_with_matches shows file paths, count shows match counts",
                        "default": "files_with_matches"
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of context lines around matches (only for content mode)",
                        "minimum": 0,
                        "maximum": 10,
                        "default": 0
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 100
                    },
                    "show_line_numbers": {
                        "type": "boolean",
                        "description": "Show line numbers in content mode",
                        "default": True
                    }
                },
                "required": ["pattern"],
                "additionalProperties": False
            }
        }
    
    async def execute(self, parameters: Dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        """Execute grep search."""
        pattern = parameters["pattern"]
        path = parameters.get("path", ".")
        glob_pattern = parameters.get("glob")
        case_insensitive = parameters.get("case_insensitive", False)
        output_mode = parameters.get("output_mode", "files_with_matches")
        context_lines = parameters.get("context_lines", 0)
        max_results = parameters.get("max_results", 100)
        show_line_numbers = parameters.get("show_line_numbers", True)
        
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
                return ToolResult.error(f"Path does not exist: {path}")
            
            # Compile regex pattern
            regex_flags = re.IGNORECASE if case_insensitive else 0
            try:
                compiled_pattern = re.compile(pattern, regex_flags)
            except re.error as e:
                return ToolResult.error(f"Invalid regex pattern: {e}")
            
            # Find files to search
            files_to_search = self._find_files(path, glob_pattern)
            
            if not files_to_search:
                return ToolResult.error(f"No files found to search in {path}")
            
            # Perform search
            results = []
            total_matches = 0
            
            for file_path in files_to_search:
                if len(results) >= max_results:
                    break
                    
                file_matches = self._search_file(
                    file_path, compiled_pattern, output_mode, 
                    context_lines, show_line_numbers
                )
                
                if file_matches["matches"]:
                    results.append(file_matches)
                    total_matches += file_matches["match_count"]
            
            # Format output based on mode
            if output_mode == "files_with_matches":
                output = "\\n".join([result["file_path"] for result in results])
            elif output_mode == "count":
                count_lines = []
                for result in results:
                    count_lines.append(f"{result['match_count']}:{result['file_path']}")
                output = "\\n".join(count_lines)
            else:  # content mode
                content_lines = []
                for result in results:
                    content_lines.append(f"\\n=== {result['file_path']} ===")
                    content_lines.extend(result["matches"])
                output = "\\n".join(content_lines)
            
            metadata = {
                "pattern": pattern,
                "search_path": path,
                "files_searched": len(files_to_search),
                "files_with_matches": len(results),
                "total_matches": total_matches,
                "output_mode": output_mode,
                "case_insensitive": case_insensitive,
                "truncated": len(results) >= max_results
            }
            
            if not results:
                return ToolResult.success("No matches found", metadata)
            
            return ToolResult.success(output, metadata)
            
        except Exception as e:
            return ToolResult.error(f"Error during search: {str(e)}")
    
    def _find_files(self, path: str, glob_pattern: Optional[str]) -> List[str]:
        """Find files to search based on path and glob pattern."""
        files = []
        
        if os.path.isfile(path):
            # Single file
            files.append(path)
        else:
            # Directory - walk and find files
            for root, dirs, filenames in os.walk(path):
                # Skip hidden directories and common ignore patterns  
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'node_modules', '__pycache__', '.git'}]
                
                for filename in filenames:
                    if filename.startswith('.'):
                        continue
                    
                    file_path = os.path.join(root, filename)
                    
                    # Apply glob filter
                    if glob_pattern:
                        if not fnmatch.fnmatch(filename, glob_pattern):
                            continue
                    
                    # Skip binary files
                    if self._is_binary_file(file_path):
                        continue
                    
                    files.append(file_path)
        
        return files[:1000]  # Limit to prevent excessive searching
    
    def _search_file(self, file_path: str, pattern: re.Pattern, output_mode: str, 
                    context_lines: int, show_line_numbers: bool) -> Dict[str, Any]:
        """Search a single file for the pattern."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            matches = []
            match_count = 0
            
            for line_num, line in enumerate(lines, 1):
                line = line.rstrip('\\n')
                match = pattern.search(line)
                
                if match:
                    match_count += 1
                    
                    if output_mode == "content":
                        # Add context lines
                        context_start = max(0, line_num - 1 - context_lines)
                        context_end = min(len(lines), line_num + context_lines)
                        
                        context_matches = []
                        for i in range(context_start, context_end):
                            context_line = lines[i].rstrip('\\n')
                            prefix = ""
                            
                            if show_line_numbers:
                                prefix = f"{i+1:4}: "
                            
                            if i == line_num - 1:  # The actual match
                                context_matches.append(f"{prefix}> {context_line}")
                            else:
                                context_matches.append(f"{prefix}  {context_line}")
                        
                        matches.extend(context_matches)
                        if context_lines > 0:
                            matches.append("--")  # Separator
            
            return {
                "file_path": file_path,
                "matches": matches,
                "match_count": match_count
            }
            
        except Exception:
            # Skip files that can't be read
            return {
                "file_path": file_path,
                "matches": [],
                "match_count": 0
            }