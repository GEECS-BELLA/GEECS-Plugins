"""Dependency-direction pin: geecs_bluesky never imports geecs_scanner.

The BlueskyScanner bridge and its delegation seam died with the
queueserver migration (W5, issue #649) — the ScanRequest engine surface
is pinned in ``tests/test_scan_request_runner.py`` — but this AST-level
guard outlives it: the optimization loader stays injected and the
relocated Xopt/evaluator stack must never grow a ``geecs_scanner``
import.
"""

from __future__ import annotations

from pathlib import Path


def test_dependency_direction_no_geecs_scanner_import() -> None:
    """geecs_bluesky must never import geecs_scanner (the loader stays
    injected; the Xopt/evaluator stack lives in geecs_bluesky.optimization).
    AST-level check so docstring usage examples don't count — only real
    import statements."""
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
