"""Dependency-direction pin: geecs_bluesky never imports geecs_scanner.

The web scanner (``GeecsScanner``) depends on this package — the queue
client, the import-light contract modules — and the edge is one-way.
An AST-level guard, so a docstring mentioning the name does not count.
"""

from __future__ import annotations

from pathlib import Path


def test_dependency_direction_no_geecs_scanner_import() -> None:
    """geecs_bluesky must never import geecs_scanner — real import statements only."""
    import ast

    import geecs_bluesky

    def _imports_geecs_scanner(tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name.split(".")[0] == "geecs_scanner" for a in node.names):
                    return True
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] == "geecs_scanner":
                    return True
        return False

    package_root = Path(geecs_bluesky.__file__).parent
    offenders = [
        str(path.relative_to(package_root))
        for path in sorted(package_root.rglob("*.py"))
        if _imports_geecs_scanner(ast.parse(path.read_text()))
    ]
    assert offenders == []
