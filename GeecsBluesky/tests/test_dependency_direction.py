"""geecs_bluesky never imports geecs_scanner under ``TYPE_CHECKING``.

Only the ``TYPE_CHECKING`` case: ``.importlinter`` (bluesky-edges) excludes those imports and covers the rest.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _type_checking_imports_of_scanner(tree: ast.AST) -> bool:
    for block in ast.walk(tree):
        if not (isinstance(block, ast.If) and _is_type_checking(block.test)):
            continue
        for stmt in block.body:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Import):
                    if any(a.name.split(".")[0] == "geecs_scanner" for a in node.names):
                        return True
                elif isinstance(node, ast.ImportFrom):
                    if (node.module or "").split(".")[0] == "geecs_scanner":
                        return True
    return False


def test_no_type_checking_import_of_geecs_scanner() -> None:
    """No ``if TYPE_CHECKING:`` block in geecs_bluesky imports geecs_scanner."""
    import geecs_bluesky

    package_root = Path(geecs_bluesky.__file__).parent
    offenders = [
        str(path.relative_to(package_root))
        for path in sorted(package_root.rglob("*.py"))
        if _type_checking_imports_of_scanner(ast.parse(path.read_text()))
    ]
    assert offenders == []
