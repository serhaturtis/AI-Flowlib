"""
Provider contract enforcement test.

This test ensures that ALL providers decorated with @provider in the codebase explicitly pass the 'settings_class' argument, enforcing fast-fail contract compliance.

This is NOT a runtime test of provider logic, but a static code compliance check.
"""
import ast
import os
import pytest

PROVIDERS_DIR = os.path.dirname(__file__)

# List of subdirs to scan for providers
SUBDIRS = [
    "embedding",
    "llm",
    "storage",
    "cache",
    "db",
    "vector",
    "graph",
    "mq",
    "gpu",
]

def find_provider_files():
    files = []
    for subdir in SUBDIRS:
        dir_path = os.path.join(PROVIDERS_DIR, subdir)
        if not os.path.isdir(dir_path):
            continue
        for fname in os.listdir(dir_path):
            if fname.endswith(".py") and not fname.startswith("__"):
                files.append(os.path.join(dir_path, fname))
    return files

@pytest.mark.parametrize("file_path", find_provider_files())
def test_provider_decorator_contract(file_path):
    """Ensure all @provider decorators have settings_class argument."""
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=file_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for deco in node.decorator_list:
                if isinstance(deco, ast.Call) and getattr(deco.func, "id", None) == "provider":
                    # Check for settings_class in keywords
                    kwarg_names = {kw.arg for kw in deco.keywords}
                    assert "settings_class" in kwarg_names, (
                        f"Provider class '{node.name}' in {file_path} is missing 'settings_class' in @provider decorator. "
                        "Contract compliance required."
                    )
