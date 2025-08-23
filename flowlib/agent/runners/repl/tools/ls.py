"""List tool - opencode style implementation with flowlib conventions."""

import os
import stat
import time
from typing import Dict, Any, List, Optional
from .base import REPLTool, ToolResult, ToolExecutionContext


class ListTool(REPLTool):
    """List directory contents with detailed information."""
    
    async def init(self) -> Dict[str, Any]:
        """Initialize list tool schema."""
        return {
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list (default: current directory)",
                        "default": "."
                    },
                    "show_hidden": {
                        "type": "boolean",
                        "description": "Show hidden files and directories",
                        "default": False
                    },
                    "long_format": {
                        "type": "boolean", 
                        "description": "Show detailed information (permissions, size, date)",
                        "default": False
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["name", "mtime", "size", "type"],
                        "description": "Sort by name, modification time, size, or type",
                        "default": "name"
                    },
                    "reverse": {
                        "type": "boolean",
                        "description": "Reverse sort order",
                        "default": False
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "List directories recursively",
                        "default": False
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth for recursive listing",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 2
                    }
                },
                "additionalProperties": False
            }
        }
    
    async def execute(self, parameters: Dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        """Execute list operation."""
        path = parameters.get("path", ".")
        show_hidden = parameters.get("show_hidden", False)
        long_format = parameters.get("long_format", False)
        sort_by = parameters.get("sort_by", "name")
        reverse = parameters.get("reverse", False)
        recursive = parameters.get("recursive", False)
        max_depth = parameters.get("max_depth", 2)
        
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
            
            # Handle single file
            if os.path.isfile(path):
                file_info = self._get_file_info(path, long_format)
                return ToolResult.success(file_info["formatted"], {"path": path, "type": "file"})
            
            # List directory
            if recursive:
                output, metadata = self._list_recursive(path, show_hidden, long_format, sort_by, reverse, max_depth)
            else:
                output, metadata = self._list_directory(path, show_hidden, long_format, sort_by, reverse)
            
            metadata["path"] = path
            return ToolResult.success(output, metadata)
            
        except PermissionError:
            return ToolResult.error(f"Permission denied accessing: {path}")
        except Exception as e:
            return ToolResult.error(f"Error listing directory: {str(e)}")
    
    def _list_directory(self, path: str, show_hidden: bool, long_format: bool, 
                       sort_by: str, reverse: bool) -> tuple[str, Dict[str, Any]]:
        """List a single directory."""
        entries = []
        
        try:
            for entry_name in os.listdir(path):
                if not show_hidden and entry_name.startswith('.'):
                    continue
                
                entry_path = os.path.join(path, entry_name)
                file_info = self._get_file_info(entry_path, long_format)
                if file_info:
                    entries.append(file_info)
        
        except PermissionError:
            return f"Permission denied reading directory: {path}", {"error": True}
        
        # Sort entries
        entries = self._sort_entries(entries, sort_by, reverse)
        
        # Format output
        if long_format:
            output_lines = [entry["formatted"] for entry in entries]
            output = "\\n".join(output_lines)
        else:
            # Simple format - just names with type indicators
            names = []
            for entry in entries:
                name = entry["name"]
                if entry["is_dir"]:
                    name += "/"
                elif entry["is_executable"]:
                    name += "*"
                elif entry["is_link"]:
                    name += "@"
                names.append(name)
            output = "\\n".join(names)
        
        metadata = {
            "total_entries": len(entries),
            "directories": sum(1 for e in entries if e["is_dir"]),
            "files": sum(1 for e in entries if not e["is_dir"]),
            "long_format": long_format
        }
        
        return output, metadata
    
    def _list_recursive(self, path: str, show_hidden: bool, long_format: bool,
                       sort_by: str, reverse: bool, max_depth: int) -> tuple[str, Dict[str, Any]]:
        """List directories recursively."""
        all_entries = []
        total_dirs = 0
        total_files = 0
        
        def walk_directory(current_path: str, depth: int):
            nonlocal total_dirs, total_files
            
            if depth > max_depth:
                return
            
            try:
                entries = []
                for entry_name in os.listdir(current_path):
                    if not show_hidden and entry_name.startswith('.'):
                        continue
                    
                    entry_path = os.path.join(current_path, entry_name)
                    file_info = self._get_file_info(entry_path, long_format)
                    if file_info:
                        # Add relative path for recursive listing
                        rel_path = os.path.relpath(entry_path, path)
                        file_info["rel_path"] = rel_path
                        entries.append(file_info)
                        
                        if file_info["is_dir"]:
                            total_dirs += 1
                            walk_directory(entry_path, depth + 1)
                        else:
                            total_files += 1
                
                # Sort and add to all_entries
                entries = self._sort_entries(entries, sort_by, reverse)
                all_entries.extend(entries)
                
            except PermissionError:
                # Skip directories we can't read
                pass
        
        walk_directory(path, 0)
        
        # Format output
        if long_format:
            output_lines = []
            current_dir = ""
            for entry in all_entries:
                entry_dir = os.path.dirname(entry["rel_path"])
                if entry_dir != current_dir:
                    if output_lines:  # Not the first directory
                        output_lines.append("")
                    output_lines.append(f"{entry_dir}/:")
                    current_dir = entry_dir
                output_lines.append(entry["formatted"])
            output = "\\n".join(output_lines)
        else:
            output = "\\n".join([entry["rel_path"] for entry in all_entries])
        
        metadata = {
            "total_entries": len(all_entries),
            "directories": total_dirs,
            "files": total_files,
            "max_depth": max_depth,
            "long_format": long_format
        }
        
        return output, metadata
    
    def _get_file_info(self, path: str, long_format: bool) -> Optional[Dict[str, Any]]:
        """Get information about a file or directory."""
        try:
            stat_info = os.lstat(path)  # Use lstat to not follow symlinks
            name = os.path.basename(path)
            
            is_dir = stat.S_ISDIR(stat_info.st_mode)
            is_link = stat.S_ISLNK(stat_info.st_mode)
            is_executable = bool(stat_info.st_mode & stat.S_IXUSR)
            
            file_info = {
                "name": name,
                "path": path,
                "is_dir": is_dir,
                "is_link": is_link,
                "is_executable": is_executable,
                "size": stat_info.st_size,
                "mtime": stat_info.st_mtime
            }
            
            if long_format:
                # Format like ls -l
                perms = stat.filemode(stat_info.st_mode)
                size_str = self._format_size(stat_info.st_size)
                mtime_str = time.strftime("%b %d %H:%M", time.localtime(stat_info.st_mtime))
                
                formatted = f"{perms} {size_str:>8} {mtime_str} {name}"
                if is_link:
                    try:
                        target = os.readlink(path)
                        formatted += f" -> {target}"
                    except OSError:
                        pass
            else:
                formatted = name
            
            file_info["formatted"] = formatted
            return file_info
            
        except (OSError, PermissionError):
            return None
    
    def _sort_entries(self, entries: List[Dict[str, Any]], sort_by: str, reverse: bool) -> List[Dict[str, Any]]:
        """Sort entries by specified criteria."""
        if sort_by == "name":
            entries.sort(key=lambda x: x["name"].lower())
        elif sort_by == "mtime":
            entries.sort(key=lambda x: x["mtime"])
        elif sort_by == "size":
            entries.sort(key=lambda x: x["size"])
        elif sort_by == "type":
            # Directories first, then by name
            entries.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
        
        if reverse:
            entries.reverse()
        
        return entries
    
    def _format_size(self, size: int) -> str:
        """Format file size in human readable format."""
        if size < 1024:
            return f"{size}B"
        elif size < 1024 * 1024:
            return f"{size/1024:.1f}K"
        elif size < 1024 * 1024 * 1024:
            return f"{size/(1024*1024):.1f}M"
        else:
            return f"{size/(1024*1024*1024):.1f}G"