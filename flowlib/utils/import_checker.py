import importlib
import os
import sys
import traceback
from typing import List


def find_python_modules(base_dir: str, base_package: str) -> List[str]:
    """
    Recursively find all Python modules in base_dir, returning their import paths.
    Skips __pycache__ and hidden directories.
    Includes __init__.py files.
    """
    modules = []
    for root, dirs, files in os.walk(base_dir):
        # Skip __pycache__ and hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for file in files:
            if file.endswith(".py"):
                rel_dir = os.path.relpath(root, base_dir)
                if rel_dir == ".":
                    rel_dir = ""
                module_name = file[:-3]  # strip .py
                parts = [base_package]
                if rel_dir:
                    parts += rel_dir.split(os.sep)
                if module_name != "__init__":
                    parts.append(module_name)
                # For __init__.py, just use the package path
                import_path = ".".join(parts)
                modules.append(import_path)
    return modules


def main() -> None:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_package = "flowlib"
    sys.path.insert(0, os.path.dirname(base_dir))  # Ensure parent dir is in sys.path

    modules = find_python_modules(base_dir, base_package)
    errors = []
    print(f"Checking {len(modules)} modules for import errors...\n")
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception:
            errors.append((mod, traceback.format_exc()))
            print(f"[IMPORT ERROR] {mod}\n{traceback.format_exc()}\n")
    print("\n==================== SUMMARY ====================")
    if errors:
        print(f"{len(errors)} modules failed to import:\n")
        for mod, tb in errors:
            print(f"- {mod}")
        print("\nSee above for full tracebacks.")
    else:
        print("All modules imported successfully!")

if __name__ == "__main__":
    main()
