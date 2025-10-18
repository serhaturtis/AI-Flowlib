import ast
import importlib.util
import os
import sys
from typing import Generator, List, Optional, Tuple

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(base_dir))  # Ensure project root is in sys.path


def find_python_files(root: str) -> Generator[str, None, None]:
    for dirpath, _, files in os.walk(root):
        for file in files:
            if file.endswith('.py'):
                yield os.path.join(dirpath, file)

def search_codebase_for_symbol(root: str, symbol: str) -> List[str]:
    matches = []
    for dirpath, dirs, files in os.walk(root):
        for file in files:
            if file.endswith('.py') and file[:-3] == symbol:
                matches.append(os.path.join(dirpath, file))
        for d in dirs:
            if d == symbol:
                matches.append(os.path.join(dirpath, d))
    return matches

def resolve_mod_name(mod: str) -> Optional[str]:
    # Only prefix if not already absolute or a known external package
    if not mod:
        return mod
    if mod.startswith("flowlib.") or mod.startswith("tests.") or mod.startswith("pytest.") or mod.startswith("os") or mod.startswith("sys") or mod.startswith("json") or mod.startswith("importlib"):
        return mod
    # If it's a relative or bare import, prefix with flowlib
    return f"flowlib.{mod}"

def check_imports_in_file(filepath: str, root: str) -> List[Tuple[str, int, str, List[str], Optional[str]]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except Exception as e:
            return [(filepath, 0, f"[PARSE ERROR] {e}", [], None)]
    broken: List[Tuple[str, int, str, List[str], Optional[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                resolved_mod = resolve_mod_name(mod)
                try:
                    found = importlib.util.find_spec(resolved_mod) if resolved_mod else None
                except Exception:
                    found = None
                if not found:
                    suggestions = search_codebase_for_symbol(root, mod.split('.')[-1])
                    broken.append((filepath, node.lineno, f"import {mod}", suggestions, mod))
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue  # Skip relative imports without module
            mod = node.module
            resolved_mod = resolve_mod_name(mod)
            try:
                found = importlib.util.find_spec(resolved_mod) if resolved_mod else None
            except Exception:
                found = None
            if mod and not found:
                suggestions = search_codebase_for_symbol(root, mod.split('.')[-1])
                broken.append((filepath, node.lineno, f"from {mod} import ...", suggestions, mod))
    return broken

def main(root: str) -> None:
    all_broken = []
    for path in find_python_files(root):
        all_broken.extend(check_imports_in_file(path, root))
    for filepath, lineno, stmt, suggestions, mod in all_broken:
        print(f"[BROKEN IMPORT] {filepath}:{lineno}: {stmt}")
        if suggestions:
            print(f"  Suggestions for '{mod}':")
            for s in suggestions:
                print(f"    {s}")
        else:
            print("  No suggestions found.")
    print(f"\nTotal broken imports: {len(all_broken)}")

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "flowlib"
    main(root)
