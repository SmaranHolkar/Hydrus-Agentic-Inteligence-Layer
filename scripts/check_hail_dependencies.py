import os
import ast
import sys

ALLOWED_THIRD_PARTY = {"numpy", "cryptography", "hail_core"}

def get_stdlib_modules():
    if hasattr(sys, "stdlib_module_names"):
        return sys.stdlib_module_names
    # Fallback static list for older python versions
    return {
        "os", "sys", "time", "math", "json", "hashlib", "hmac", "collections",
        "typing", "logging", "threading", "contextlib", "io", "pathlib",
        "dataclasses", "abc", "weakref", "copy", "uuid", "enum", "struct",
        "ast", "importlib", "warnings"
    }

def check_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except Exception as e:
            return [(0, f"Parse error: {e}")]

    errors = []
    stdlib = get_stdlib_modules()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level = alias.name.split(".")[0]
                if top_level not in ALLOWED_THIRD_PARTY and top_level not in stdlib:
                    errors.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                # Relative import, always allowed within package
                continue
            if node.module:
                top_level = node.module.split(".")[0]
                if top_level not in ALLOWED_THIRD_PARTY and top_level not in stdlib:
                    errors.append((node.lineno, f"from {node.module} import ..."))
    return errors

def main():
    # Use workspace-relative path dynamically resolved or absolute
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    package_dir = os.path.join(base_dir, "src", "hail_core")
    
    all_errors = {}
    for root, _, files in os.walk(package_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                errors = check_file(path)
                if errors:
                    all_errors[path] = errors

    if all_errors:
        print("Dependency isolation audit FAILED!")
        for path, errors in all_errors.items():
            print(f"\nFile: {path}")
            for line, code in errors:
                print(f"  Line {line}: Forbidden import '{code}' found.")
        sys.exit(1)
    else:
        print("Dependency isolation audit PASSED! No coupled imports found under src/hail_core/.")
        sys.exit(0)

if __name__ == "__main__":
    main()
