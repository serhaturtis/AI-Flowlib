import os
import sys
import json
from typing import Dict, List


def replace_in_file(filepath: str, mapping: Dict[str, str]) -> bool:
    changed = False
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        original = line
        for old, new in mapping.items():
            if old in line:
                line = line.replace(old, new)
        if line != original:
            changed = True
        new_lines.append(line)
    if changed:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
    return changed

def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python replace_imports.py <mapping.json> <target_dir>")
        sys.exit(1)
    mapping_file = sys.argv[1]
    target_dir = sys.argv[2]
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    changed_files: List[str] = []
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                if replace_in_file(path, mapping):
                    changed_files.append(path)
    print(f"Updated {len(changed_files)} files:")
    for filepath in changed_files:
        print(f"  {filepath}")

if __name__ == "__main__":
    main() 